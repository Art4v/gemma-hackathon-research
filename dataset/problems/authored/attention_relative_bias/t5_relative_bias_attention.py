"""T5-style bucketed relative position bias -> bidirectional softmax attention (QK^T + bias[bucket(j-i)], softmax, PV)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 't5_relative_bias_attention',
    'batch': 16,
    'num_heads': 16,
    'seq_len': 1024,
    'head_dim': 64,
    'num_buckets': 32,
    'max_distance': 128,
    'bidirectional': True,
    'scale': 0.125,
}


def create_inputs(dtype=jnp.bfloat16):
    B = CONFIG['batch']
    H = CONFIG['num_heads']
    S = CONFIG['seq_len']
    D = CONFIG['head_dim']
    NB = CONFIG['num_buckets']
    key = jax.random.key(0)
    kq, kk, kv, kb = jax.random.split(key, 4)
    q = jax.random.normal(kq, (B, H, S, D), dtype)
    k = jax.random.normal(kk, (B, H, S, D), dtype)
    v = jax.random.normal(kv, (B, H, S, D), dtype)
    rel_bias = jax.random.normal(kb, (NB, H), dtype) * 0.02
    return q, k, v, rel_bias


def _relative_position_bucket(relative_position, bidirectional, num_buckets, max_distance):
    # relative_position = key_pos - query_pos (int32)
    ret = jnp.zeros_like(relative_position)
    n = -relative_position
    if bidirectional:
        num_buckets = num_buckets // 2
        ret = ret + (n < 0).astype(jnp.int32) * num_buckets
        n = jnp.abs(n)
    else:
        n = jnp.maximum(n, 0)
    max_exact = num_buckets // 2
    is_small = n < max_exact
    n_safe = jnp.maximum(n, 1).astype(jnp.float32)
    val_if_large = max_exact + (
        jnp.log(n_safe / max_exact)
        / jnp.log(max_distance / max_exact)
        * (num_buckets - max_exact)
    ).astype(jnp.int32)
    val_if_large = jnp.minimum(val_if_large, num_buckets - 1)
    ret = ret + jnp.where(is_small, n, val_if_large)
    return ret


def workload(q, k, v, rel_bias):
    S_q = q.shape[2]
    S_k = k.shape[2]
    scale = CONFIG['scale']

    q_pos = jnp.arange(S_q, dtype=jnp.int32)[:, None]
    k_pos = jnp.arange(S_k, dtype=jnp.int32)[None, :]
    buckets = _relative_position_bucket(
        k_pos - q_pos,
        CONFIG['bidirectional'],
        CONFIG['num_buckets'],
        CONFIG['max_distance'],
    )  # [S_q, S_k]
    bias = jnp.take(rel_bias.astype(jnp.float32), buckets, axis=0)  # [S_q, S_k, H]
    bias = jnp.transpose(bias, (2, 0, 1))[None]  # [1, H, S_q, S_k]

    logits = jnp.einsum('bhqd,bhkd->bhqk', q, k,
                        preferred_element_type=jnp.float32) * scale
    logits = logits + bias
    m = jnp.max(logits, axis=-1, keepdims=True)
    p = jnp.exp(logits - m)
    l = jnp.sum(p, axis=-1, keepdims=True)
    out = jnp.einsum('bhqk,bhkd->bhqd', p.astype(v.dtype), v,
                     preferred_element_type=jnp.float32)
    out = out / l
    return out.astype(q.dtype)
