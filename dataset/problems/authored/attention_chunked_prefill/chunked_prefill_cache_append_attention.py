"""Chunked prefill: append chunk K/V into KV cache at per-sequence offset, then offset-causal MHA of chunk queries over prefix+chunk."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'chunked_prefill_cache_append_attention',
    'batch': 2,
    'chunk_len': 512,
    'max_seq_len': 4096,
    'num_heads': 32,
    'head_dim': 128,
    'sm_scale': 128 ** -0.5,
    'mask_value': -1e30,
}


def create_inputs(dtype=jnp.float32):
    B = CONFIG['batch']
    C = CONFIG['chunk_len']
    S = CONFIG['max_seq_len']
    H = CONFIG['num_heads']
    D = CONFIG['head_dim']
    key = jax.random.key(0)
    kq, kk, kv, kkc, kvc, kl = jax.random.split(key, 6)
    q = jax.random.normal(kq, (B, C, H, D), dtype)
    k_new = jax.random.normal(kk, (B, C, H, D), dtype)
    v_new = jax.random.normal(kv, (B, C, H, D), dtype)
    k_cache = jax.random.normal(kkc, (B, S, H, D), dtype)
    v_cache = jax.random.normal(kvc, (B, S, H, D), dtype)
    prefix_lens = jax.random.randint(kl, (B,), 0, S - C + 1, dtype=jnp.int32)
    return q, k_new, v_new, k_cache, v_cache, prefix_lens


def workload(q, k_new, v_new, k_cache, v_cache, prefix_lens):
    B, C, H, D = q.shape
    S = k_cache.shape[1]
    scale = CONFIG['sm_scale']
    mask_value = CONFIG['mask_value']

    def _append(cache, new, off):
        return lax.dynamic_update_slice(cache, new.astype(cache.dtype), (off, 0, 0))

    k_cache = jax.vmap(_append)(k_cache, k_new, prefix_lens)
    v_cache = jax.vmap(_append)(v_cache, v_new, prefix_lens)

    s = jnp.einsum('bqhd,bkhd->bhqk', q, k_cache,
                   preferred_element_type=jnp.float32) * scale

    q_pos = prefix_lens[:, None] + jnp.arange(C, dtype=jnp.int32)[None, :]  # [B, C]
    k_pos = jnp.arange(S, dtype=jnp.int32)  # [S]
    mask = k_pos[None, None, :] <= q_pos[:, :, None]  # [B, C, S]
    s = jnp.where(mask[:, None, :, :], s, mask_value)

    m = jnp.max(s, axis=-1, keepdims=True)
    p = jnp.exp(s - m)
    l = jnp.sum(p, axis=-1, keepdims=True)
    o = jnp.einsum('bhqk,bkhd->bqhd', p, v_cache.astype(jnp.float32),
                   preferred_element_type=jnp.float32)
    o = o / jnp.transpose(l, (0, 2, 1, 3))
    return o.astype(q.dtype), k_cache, v_cache
