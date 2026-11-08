"""T5-style multi-head attention with logarithmic-bucketed relative position bias."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 't5_bucketed_relative_attention',
    'batch_size': 8,
    'num_heads': 32,
    'seq_len': 2048,
    'head_dim': 128,
    'num_buckets': 32,
    'max_distance': 128,
    'bidirectional': True,
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    k_q, k_k, k_v, k_bias = jax.random.split(key, 4)
    B = CONFIG['batch_size']
    H = CONFIG['num_heads']
    S = CONFIG['seq_len']
    D = CONFIG['head_dim']
    num_buckets = CONFIG['num_buckets']

    q = jax.random.normal(k_q, (B, H, S, D), dtype=dtype)
    k = jax.random.normal(k_k, (B, H, S, D), dtype=dtype)
    v = jax.random.normal(k_v, (B, H, S, D), dtype=dtype)
    rel_embedding = jax.random.normal(k_bias, (num_buckets, H), dtype=dtype) * 0.02
    return q, k, v, rel_embedding


def _relative_position_bucket(relative_position, bidirectional, num_buckets, max_distance):
    ret = jnp.zeros_like(relative_position)
    n = -relative_position
    if bidirectional:
        half_buckets = num_buckets // 2
        ret = ret + (n < 0).astype(jnp.int32) * half_buckets
        n = jnp.abs(n)
    else:
        half_buckets = num_buckets
        n = jnp.maximum(n, 0)

    max_exact = half_buckets // 2
    is_small = n < max_exact

    n_float = jnp.maximum(n.astype(jnp.float32), 1.0)
    max_exact_float = float(max_exact)
    log_ratio = jnp.log(n_float / max_exact_float) / jnp.log(float(max_distance) / max_exact_float)
    val_if_large = max_exact + (log_ratio * float(half_buckets - max_exact)).astype(jnp.int32)
    val_if_large = jnp.minimum(val_if_large, half_buckets - 1)

    ret = ret + jnp.where(is_small, n, val_if_large)
    return ret


def workload(q, k, v, rel_embedding):
    S = q.shape[2]
    D = q.shape[3]
    num_buckets = CONFIG['num_buckets']
    max_distance = CONFIG['max_distance']
    bidirectional = CONFIG['bidirectional']

    q_idx = jnp.arange(S, dtype=jnp.int32)[:, None]
    k_idx = jnp.arange(S, dtype=jnp.int32)[None, :]
    relative_position = k_idx - q_idx

    rp_bucket = _relative_position_bucket(
        relative_position,
        bidirectional=bidirectional,
        num_buckets=num_buckets,
        max_distance=max_distance,
    )

    bias = rel_embedding[rp_bucket]
    bias = jnp.transpose(bias, (2, 0, 1))[None, :, :, :].astype(jnp.float32)

    scale = 1.0 / jnp.sqrt(float(D))
    scores = jnp.einsum('bhsd,bhtd->bhst', q.astype(jnp.float32), k.astype(jnp.float32)) * scale
    scores = scores + bias

    attn_weights = jax.nn.softmax(scores, axis=-1)
    out = jnp.einsum('bhst,bhtd->bhsd', attn_weights, v.astype(jnp.float32))
    return out.astype(q.dtype)
