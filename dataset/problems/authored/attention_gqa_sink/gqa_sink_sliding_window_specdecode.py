"""Speculative-decode GQA attention (6:1) over a ragged KV cache with sliding window + learned per-head attention sinks."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'gqa_sink_sliding_window_specdecode',
    'batch': 32,
    'num_query_tokens': 4,
    'max_cache_len': 8192,
    'num_q_heads': 48,
    'num_kv_heads': 8,
    'head_dim': 64,
    'window': 1024,
    'weight_scale': 0.02,
}


def create_inputs(dtype=jnp.bfloat16):
    c = CONFIG
    B, T, S = c['batch'], c['num_query_tokens'], c['max_cache_len']
    Hq, Hkv, D = c['num_q_heads'], c['num_kv_heads'], c['head_dim']
    kq, kk, kv, ks, kl = jax.random.split(jax.random.key(0), 5)
    q = jax.random.normal(kq, (B, T, Hq, D), dtype)
    k_cache = jax.random.normal(kk, (B, S, Hkv, D), dtype)
    v_cache = jax.random.normal(kv, (B, S, Hkv, D), dtype)
    sinks = jax.random.normal(ks, (Hq,), jnp.float32) * c['weight_scale']
    lengths = jax.random.randint(kl, (B,), c['window'], S + 1, dtype=jnp.int32)
    return q, k_cache, v_cache, sinks, lengths


def workload(q, k_cache, v_cache, sinks, lengths):
    B, T, Hq, D = q.shape
    S, Hkv = k_cache.shape[1], k_cache.shape[2]
    G = Hq // Hkv
    window = CONFIG['window']
    scale = 1.0 / jnp.sqrt(jnp.float32(D))

    qg = q.reshape(B, T, Hkv, G, D)
    # scores: (B, Hkv, G, T, S) in float32
    s = jnp.einsum('btkgd,bskd->bkgts', qg, k_cache,
                   preferred_element_type=jnp.float32) * scale

    qpos = lengths[:, None] - T + jnp.arange(T, dtype=jnp.int32)[None, :]  # (B, T)
    kpos = jnp.arange(S, dtype=jnp.int32)  # (S,)
    valid = ((kpos[None, None, :] <= qpos[:, :, None])
             & (kpos[None, None, :] > qpos[:, :, None] - window)
             & (kpos[None, None, :] < lengths[:, None, None]))  # (B, T, S)
    valid = valid[:, None, None, :, :]  # (B, 1, 1, T, S)

    neg = jnp.float32(-1e30)
    s = jnp.where(valid, s, neg)

    sink = sinks.astype(jnp.float32).reshape(1, Hkv, G, 1, 1)
    m = jnp.maximum(jnp.max(s, axis=-1, keepdims=True), sink)
    p = jnp.where(valid, jnp.exp(s - m), 0.0)
    denom = jnp.sum(p, axis=-1, keepdims=True) + jnp.exp(sink - m)  # (B,Hkv,G,T,1)

    o = jnp.einsum('bkgts,bskd->bkgtd', p.astype(v_cache.dtype), v_cache,
                   preferred_element_type=jnp.float32)
    o = o / denom
    o = jnp.transpose(o, (0, 3, 1, 2, 4)).reshape(B, T, Hq, D)
    return o.astype(q.dtype)
