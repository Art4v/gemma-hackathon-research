"""Multi-head attention with T5-style bucketed relative position bias."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 't5_bucketed_rel_pos_attention',
    'batch_size': 4,
    'num_heads': 32,
    'seq_len': 2048,
    'head_dim': 128,
    'num_buckets': 32,
    'max_distance': 128,
    'bidirectional': True,
}


def create_inputs(dtype=jnp.bfloat16):
    k0, k1, k2, k3 = jax.random.split(jax.random.key(0), 4)
    b = CONFIG['batch_size']
    h = CONFIG['num_heads']
    s = CONFIG['seq_len']
    d = CONFIG['head_dim']
    nb = CONFIG['num_buckets']

    q = jax.random.normal(k0, (b, h, s, d), dtype=dtype)
    k = jax.random.normal(k1, (b, h, s, d), dtype=dtype)
    v = jax.random.normal(k2, (b, h, s, d), dtype=dtype)
    rel_attention_bias = jax.random.normal(k3, (nb, h), dtype=dtype) * 0.02
    return q, k, v, rel_attention_bias


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
    val_if_large = max_exact + (
        jnp.log(n_float / max_exact_float)
        / jnp.log(float(max_distance) / max_exact_float)
        * float(half_buckets - max_exact)
    ).astype(jnp.int32)
    val_if_large = jnp.minimum(val_if_large, half_buckets - 1)

    ret = ret + jnp.where(is_small, n, val_if_large)
    return ret


def workload(q, k, v, rel_attention_bias):
    s = q.shape[2]
    d = q.shape[3]
    num_buckets = CONFIG['num_buckets']
    max_distance = CONFIG['max_distance']
    bidirectional = CONFIG['bidirectional']

    q_idx = jnp.arange(s, dtype=jnp.int32)[:, None]
    k_idx = jnp.arange(s, dtype=jnp.int32)[None, :]
    relative_position = k_idx - q_idx

    rp_bucket = _relative_position_bucket(
        relative_position,
        bidirectional=bidirectional,
        num_buckets=num_buckets,
        max_distance=max_distance,
    )

    values = rel_attention_bias[rp_bucket]
    position_bias = jnp.transpose(values, (2, 0, 1))[None, :, :, :].astype(jnp.float32)

    scale = 1.0 / jnp.sqrt(float(d))
    logits = jnp.einsum('bhqd,bhkd->bhqk', q.astype(jnp.float32), k.astype(jnp.float32)) * scale
    logits = logits + position_bias

    weights = jax.nn.softmax(logits, axis=-1)
    out = jnp.einsum('bhqk,bhkd->bhqd', weights, v.astype(jnp.float32))
    return out.astype(q.dtype)
