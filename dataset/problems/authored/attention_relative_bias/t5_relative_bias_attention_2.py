"""T5-style bucketed relative position bias attention: (scale*QK^T + table[bucket(j-i)]) -> softmax -> PV."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 't5_relative_bias_attention',
    'batch': 2,
    'num_heads': 64,
    'seq_len': 2048,
    'head_dim': 64,
    'num_buckets': 32,
    'max_distance': 128,
    'bidirectional': True,
    'scale': 0.125,
    'bias_std': 1.0,
}


def create_inputs(dtype=jnp.float32):
    B, H, S, D = CONFIG['batch'], CONFIG['num_heads'], CONFIG['seq_len'], CONFIG['head_dim']
    key = jax.random.key(0)
    kq, kk, kv, kb = jax.random.split(key, 4)
    q = jax.random.normal(kq, (B, H, S, D), dtype)
    k = jax.random.normal(kk, (B, H, S, D), dtype)
    v = jax.random.normal(kv, (B, H, S, D), dtype)
    bias_table = jax.random.normal(kb, (CONFIG['num_buckets'], H), dtype) * CONFIG['bias_std']
    return q, k, v, bias_table


def _relative_position_bucket(relative_position):
    num_buckets = CONFIG['num_buckets']
    max_distance = CONFIG['max_distance']
    buckets = jnp.zeros_like(relative_position)
    if CONFIG['bidirectional']:
        num_buckets = num_buckets // 2
        buckets = buckets + (relative_position > 0).astype(jnp.int32) * num_buckets
        rel = jnp.abs(relative_position)
    else:
        rel = -jnp.minimum(relative_position, 0)
    max_exact = num_buckets // 2
    is_small = rel < max_exact
    rel_f = jnp.maximum(rel, 1).astype(jnp.float32)
    large = max_exact + (
        jnp.log(rel_f / max_exact) / jnp.log(max_distance / max_exact) * (num_buckets - max_exact)
    ).astype(jnp.int32)
    large = jnp.minimum(large, num_buckets - 1)
    return buckets + jnp.where(is_small, rel, large)


def workload(q, k, v, bias_table):
    S_q = q.shape[2]
    S_k = k.shape[2]
    q_pos = jnp.arange(S_q, dtype=jnp.int32)[:, None]
    k_pos = jnp.arange(S_k, dtype=jnp.int32)[None, :]
    buckets = _relative_position_bucket(k_pos - q_pos)            # (Sq, Sk)
    bias = jnp.take(bias_table.astype(jnp.float32), buckets, axis=0)  # (Sq, Sk, H)
    bias = jnp.transpose(bias, (2, 0, 1))[None]                   # (1, H, Sq, Sk)

    scores = jnp.einsum('bhqd,bhkd->bhqk', q, k,
                        preferred_element_type=jnp.float32) * CONFIG['scale']
    scores = scores + bias
    scores = scores - jnp.max(scores, axis=-1, keepdims=True)
    p = jnp.exp(scores)
    p = p / jnp.sum(p, axis=-1, keepdims=True)
    out = jnp.einsum('bhqk,bhkd->bhqd', p.astype(v.dtype), v,
                     preferred_element_type=jnp.float32)
    return out.astype(q.dtype)
