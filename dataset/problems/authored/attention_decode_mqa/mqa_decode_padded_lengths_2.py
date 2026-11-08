"""Multi-query attention single-token decode over a padded KV cache with per-sequence valid lengths (QK^T -> length mask -> softmax -> PV)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'mqa_decode_padded_lengths',
    'batch': 32,
    'num_heads': 32,
    'head_dim': 128,
    'max_seq_len': 8192,
    'min_len': 1,
    'sm_scale': 128 ** -0.5,
}


def create_inputs(dtype=jnp.bfloat16):
    B = CONFIG['batch']
    H = CONFIG['num_heads']
    D = CONFIG['head_dim']
    S = CONFIG['max_seq_len']
    kq, kk, kv, kl = jax.random.split(jax.random.key(0), 4)
    q = jax.random.normal(kq, (B, H, D), dtype)
    k_cache = jax.random.normal(kk, (B, S, D), dtype)
    v_cache = jax.random.normal(kv, (B, S, D), dtype)
    lengths = jax.random.randint(kl, (B,), CONFIG['min_len'], S + 1, dtype=jnp.int32)
    return q, k_cache, v_cache, lengths


def workload(q, k_cache, v_cache, lengths):
    S = k_cache.shape[1]
    scale = CONFIG['sm_scale']
    # Scores for all heads against the single shared KV head, f32 accumulation.
    s = jnp.einsum('bhd,bsd->bhs', q, k_cache,
                   preferred_element_type=jnp.float32) * scale
    pos = jnp.arange(S, dtype=jnp.int32)
    valid = pos[None, None, :] < lengths[:, None, None]
    s = jnp.where(valid, s, jnp.finfo(jnp.float32).min)
    m = jnp.max(s, axis=-1, keepdims=True)
    p = jnp.exp(s - m)
    p = jnp.where(valid, p, 0.0)
    l = jnp.sum(p, axis=-1, keepdims=True)
    o = jnp.einsum('bhs,bsd->bhd', p.astype(v_cache.dtype), v_cache,
                   preferred_element_type=jnp.float32)
    o = o / l
    return o.astype(q.dtype)
