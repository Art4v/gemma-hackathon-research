"""Chunked-prefill attention: KV-cache chunk write at per-sequence offset + offset-causal softmax attention."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'chunked_prefill_kv_offset_attention',
    'batch': 4,
    'num_heads': 32,
    'num_kv_heads': 32,
    'chunk_len': 512,
    'cache_len': 4096,
    'head_dim': 128,
    'sm_scale': 128 ** -0.5,
    'mask_value': -1e30,
}


def create_inputs(dtype=jnp.float32):
    B = CONFIG['batch']
    H = CONFIG['num_heads']
    Hkv = CONFIG['num_kv_heads']
    C = CONFIG['chunk_len']
    S = CONFIG['cache_len']
    D = CONFIG['head_dim']
    key = jax.random.key(0)
    kq, kk, kv, kkc, kvc, kl = jax.random.split(key, 6)
    q = jax.random.normal(kq, (B, H, C, D), dtype)
    k_new = jax.random.normal(kk, (B, Hkv, C, D), dtype)
    v_new = jax.random.normal(kv, (B, Hkv, C, D), dtype)
    k_cache = jax.random.normal(kkc, (B, Hkv, S, D), dtype)
    v_cache = jax.random.normal(kvc, (B, Hkv, S, D), dtype)
    prefix_lens = jax.random.randint(kl, (B,), 0, S - C + 1, dtype=jnp.int32)
    return q, k_new, v_new, k_cache, v_cache, prefix_lens


def workload(q, k_new, v_new, k_cache, v_cache, prefix_lens):
    B, H, C, D = q.shape
    Hkv = k_cache.shape[1]
    S = k_cache.shape[2]
    group = H // Hkv
    scale = CONFIG['sm_scale']
    mask_value = CONFIG['mask_value']

    def per_seq(args):
        qb, knb, vnb, kcb, vcb, plen = args
        # Write the new chunk into the cache at offset plen.
        kcb = lax.dynamic_update_slice(kcb, knb.astype(kcb.dtype), (0, plen, 0))
        vcb = lax.dynamic_update_slice(vcb, vnb.astype(vcb.dtype), (0, plen, 0))
        kf = jnp.repeat(kcb, group, axis=0).astype(jnp.float32)  # (H, S, D)
        vf = jnp.repeat(vcb, group, axis=0).astype(jnp.float32)
        qf = qb.astype(jnp.float32)
        s = jnp.einsum('hcd,hsd->hcs', qf, kf) * scale  # (H, C, S)
        q_pos = plen + jnp.arange(C, dtype=jnp.int32)[:, None]
        k_pos = jnp.arange(S, dtype=jnp.int32)[None, :]
        mask = k_pos <= q_pos  # (C, S)
        s = jnp.where(mask[None], s, mask_value)
        m = jnp.max(s, axis=-1, keepdims=True)
        p = jnp.exp(s - m)
        l = jnp.sum(p, axis=-1, keepdims=True)
        o = jnp.einsum('hcs,hsd->hcd', p, vf) / l
        return o.astype(qb.dtype), kcb, vcb

    out, k_upd, v_upd = lax.map(
        per_seq, (q, k_new, v_new, k_cache, v_cache, prefix_lens))
    return out, k_upd, v_upd
