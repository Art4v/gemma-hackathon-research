"""Speculative-verify GQA decode over a sliding-window ring-buffer KV cache with learned per-head attention sinks."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'gqa_sink_ring_cache_spec_verify',
    'batch': 24,
    'num_query_tokens': 4,
    'num_q_heads': 40,
    'num_kv_heads': 8,
    'head_dim': 128,
    'window_size': 4092,
    'ring_size': 4096,  # window_size + num_query_tokens
    'max_context_multiple': 3,
    'sink_scale': 1.0,
}


def create_inputs(dtype=jnp.float32):
    B = CONFIG['batch']
    T = CONFIG['num_query_tokens']
    Hq = CONFIG['num_q_heads']
    Hkv = CONFIG['num_kv_heads']
    D = CONFIG['head_dim']
    R = CONFIG['ring_size']
    k0 = jax.random.key(0)
    kq, kk, kv, kl, ks = jax.random.split(k0, 5)
    q = jax.random.normal(kq, (B, T, Hq, D), dtype)
    k_cache = jax.random.normal(kk, (B, R, Hkv, D), dtype)
    v_cache = jax.random.normal(kv, (B, R, Hkv, D), dtype)
    context_lens = jax.random.randint(
        kl, (B,), 1, CONFIG['max_context_multiple'] * R, dtype=jnp.int32)
    sinks = jax.random.normal(ks, (Hq,), jnp.float32) * CONFIG['sink_scale']
    return q, k_cache, v_cache, context_lens, sinks


def workload(q, k_cache, v_cache, context_lens, sinks):
    B, T, Hq, D = q.shape
    R = k_cache.shape[1]
    Hkv = k_cache.shape[2]
    G = Hq // Hkv
    W = CONFIG['window_size']
    scale = 1.0 / (D ** 0.5)

    # Absolute positions of query tokens: context_len + t (draft K/V already written).
    t_idx = jnp.arange(T, dtype=jnp.int32)
    q_pos = context_lens[:, None] + t_idx[None, :]                     # [B, T]
    last = context_lens + T - 1                                         # [B]
    slots = jnp.arange(R, dtype=jnp.int32)
    k_pos = last[:, None] - jnp.mod(last[:, None] - slots[None, :], R)  # [B, R]

    valid = (k_pos[:, None, :] >= 0)
    causal = k_pos[:, None, :] <= q_pos[:, :, None]
    in_win = (q_pos[:, :, None] - k_pos[:, None, :]) < W
    mask = valid & causal & in_win                                       # [B, T, R]

    qg = q.reshape(B, T, Hkv, G, D)
    s = jnp.einsum('btkgd,brkd->bkgtr', qg, k_cache,
                   preferred_element_type=jnp.float32) * scale          # [B,Hkv,G,T,R]
    s = jnp.where(mask[:, None, None, :, :], s, -jnp.inf)

    sink = sinks.astype(jnp.float32).reshape(1, Hkv, G, 1, 1)
    m = jnp.maximum(jnp.max(s, axis=-1, keepdims=True), sink)
    p = jnp.exp(s - m)
    denom = jnp.sum(p, axis=-1, keepdims=True) + jnp.exp(sink - m)
    o = jnp.einsum('bkgtr,brkd->btkgd', p.astype(v_cache.dtype), v_cache,
                   preferred_element_type=jnp.float32)
    o = o / jnp.transpose(denom, (0, 3, 1, 2, 4))                        # [B,T,Hkv,G,1]
    return o.reshape(B, T, Hq, D).astype(q.dtype)
