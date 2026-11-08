"""Multi-head attention with T5-style bucketed relative position bias."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 't5_bucketed_rel_pos_attention',
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
    k_q, k_k, k_v, k_b = jax.random.split(key, 4)
    b = CONFIG['batch_size']
    h = CONFIG['num_heads']
    s = CONFIG['seq_len']
    d = CONFIG['head_dim']
    num_buckets = CONFIG['num_buckets']

    q = jax.random.normal(k_q, (b, h, s, d), dtype=dtype)
    k = jax.random.normal(k_k, (b, h, s, d), dtype=dtype)
    v = jax.random.normal(k_v, (b, h, s, d), dtype=dtype)
    rel_attention_bias = jax.random.normal(k_b, (num_buckets, h), dtype=dtype) * 0.02
    return q, k, v, rel_attention_bias


def _relative_position_bucket(relative_position, bidirectional, num_buckets, max_distance):
    relative_buckets = jnp.zeros_like(relative_position, dtype=jnp.int32)
    if bidirectional:
        half_buckets = num_buckets // 2
        relative_buckets = relative_buckets + (relative_position > 0).astype(jnp.int32) * half_buckets
        relative_position = jnp.abs(relative_position)
        num_buckets = half_buckets
    else:
        relative_position = -jnp.minimum(relative_position, jnp.zeros_like(relative_position))

    max_exact = num_buckets // 2
    is_small = relative_position < max_exact

    rel_pos_float = relative_position.astype(jnp.float32)
    max_exact_float = float(max_exact)
    max_dist_float = float(max_distance)
    num_buckets_float = float(num_buckets)

    val_if_large = max_exact + (
        jnp.log(jnp.maximum(rel_pos_float, 1.0) / max_exact_float)
        / jnp.log(max_dist_float / max_exact_float)
        * (num_buckets_float - max_exact_float)
    ).astype(jnp.int32)
    val_if_large = jnp.minimum(val_if_large, jnp.full_like(val_if_large, num_buckets - 1))

    relative_buckets = relative_buckets + jnp.where(is_small, relative_position, val_if_large)
    return relative_buckets


def workload(q, k, v, rel_attention_bias):
    s = q.shape[2]
    num_buckets = CONFIG['num_buckets']
    max_distance = CONFIG['max_distance']
    bidirectional = CONFIG['bidirectional']

    q_pos = jnp.arange(s, dtype=jnp.int32)[:, None]
    k_pos = jnp.arange(s, dtype=jnp.int32)[None, :]
    relative_position = k_pos - q_pos

    rp_bucket = _relative_position_bucket(
        relative_position,
        bidirectional=bidirectional,
        num_buckets=num_buckets,
        max_distance=max_distance,
    )

    values = rel_attention_bias[rp_bucket]
    bias = jnp.transpose(values, (2, 0, 1))[None, :, :, :].astype(jnp.float32)

    logits = jnp.einsum('bhqd,bhkd->bhqk', q.astype(jnp.float32), k.astype(jnp.float32))
    logits = logits + bias

    weights = jax.nn.softmax(logits, axis=-1)
    out = jnp.einsum('bhqk,bhkd->bhqd', weights, v.astype(jnp.float32))
    return out.astype(q.dtype)
