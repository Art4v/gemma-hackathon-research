"""Fused RoPE on new K and contiguous KV cache append at per-sequence offsets."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'kv_rope_append',
    'batch_size': 32,
    'max_seq_len': 2048,
    'chunk_len': 64,
    'num_kv_heads': 8,
    'head_dim': 128,
    'rope_theta': 500000.0,
}


def create_inputs(dtype=jnp.bfloat16):
    b = CONFIG['batch_size']
    m = CONFIG['max_seq_len']
    c = CONFIG['chunk_len']
    h = CONFIG['num_kv_heads']
    d = CONFIG['head_dim']
    theta = CONFIG['rope_theta']

    key = jax.random.key(0)
    k1, k2, k3, k4, k5 = jax.random.split(key, 5)

    k_cache = jax.random.normal(k1, (b, m, h, d), dtype=dtype)
    v_cache = jax.random.normal(k2, (b, m, h, d), dtype=dtype)
    k_new = jax.random.normal(k3, (b, c, h, d), dtype=dtype)
    v_new = jax.random.normal(k4, (b, c, h, d), dtype=dtype)

    seq_offsets = jax.random.randint(k5, (b,), minval=0, maxval=m - c + 1, dtype=jnp.int32)

    half_d = d // 2
    freq_seq = jnp.arange(0, half_d, dtype=jnp.float32) / float(half_d)
    inv_freq = theta ** (-freq_seq)
    t = jnp.arange(m, dtype=jnp.float32)
    freqs = jnp.outer(t, inv_freq)
    cos_table = jnp.cos(freqs).astype(dtype)
    sin_table = jnp.sin(freqs).astype(dtype)

    return k_cache, v_cache, k_new, v_new, seq_offsets, cos_table, sin_table


def workload(k_cache, v_cache, k_new, v_new, seq_offsets, cos_table, sin_table):
    c = k_new.shape[1]
    d = k_new.shape[3]
    half_d = d // 2

    pos = seq_offsets[:, None] + jnp.arange(c, dtype=jnp.int32)[None, :]
    cos = cos_table[pos][:, :, None, :].astype(jnp.float32)
    sin = sin_table[pos][:, :, None, :].astype(jnp.float32)

    k_f32 = k_new.astype(jnp.float32)
    k1 = k_f32[..., :half_d]
    k2 = k_f32[..., half_d:]

    k_rot1 = k1 * cos - k2 * sin
    k_rot2 = k2 * cos + k1 * sin
    k_rot = jnp.concatenate([k_rot1, k_rot2], axis=-1).astype(k_cache.dtype)

    def update_one(kc, vc, kn, vn, offset):
        kc_out = lax.dynamic_update_slice(kc, kn, (offset, 0, 0))
        vc_out = lax.dynamic_update_slice(vc, vn, (offset, 0, 0))
        return kc_out, vc_out

    k_cache_out, v_cache_out = jax.vmap(update_one)(k_cache, v_cache, k_rot, v_new, seq_offsets)
    return k_cache_out, v_cache_out
