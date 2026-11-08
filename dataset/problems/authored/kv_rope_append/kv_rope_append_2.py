"""Fused RoPE on new K and contiguous KV cache append at per-sequence offsets."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'kv_rope_append',
    'batch_size': 32,
    'max_seq_len': 2048,
    'new_seq_len': 16,
    'num_kv_heads': 8,
    'head_dim': 128,
    'rope_theta': 500000.0,
}

def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    k1, k2, k3, k4, k5 = jax.random.split(key, 5)
    B = CONFIG['batch_size']
    M = CONFIG['max_seq_len']
    S = CONFIG['new_seq_len']
    H = CONFIG['num_kv_heads']
    D = CONFIG['head_dim']
    
    k_cache = jax.random.normal(k1, (B, M, H, D), dtype=dtype)
    v_cache = jax.random.normal(k2, (B, M, H, D), dtype=dtype)
    k_new = jax.random.normal(k3, (B, S, H, D), dtype=dtype)
    v_new = jax.random.normal(k4, (B, S, H, D), dtype=dtype)
    offsets = jax.random.randint(k5, (B,), 0, M - S, dtype=jnp.int32)
    
    return k_cache, v_cache, k_new, v_new, offsets

def workload(k_cache, v_cache, k_new, v_new, offsets):
    B, S, H, D = k_new.shape
    half_D = D // 2
    theta = CONFIG['rope_theta']
    
    freq_seq = jnp.arange(0, half_D, dtype=jnp.float32) / float(half_D)
    inv_freq = 1.0 / (theta ** freq_seq)
    
    pos = offsets[:, None] + jnp.arange(S, dtype=jnp.int32)[None, :]
    angles = pos.astype(jnp.float32)[:, :, None, None] * inv_freq[None, None, None, :]
    cos = jnp.cos(angles)
    sin = jnp.sin(angles)
    
    k_new_f32 = k_new.astype(jnp.float32)
    k1 = k_new_f32[..., :half_D]
    k2 = k_new_f32[..., half_D:]
    
    k_rot1 = k1 * cos - k2 * sin
    k_rot2 = k2 * cos + k1 * sin
    k_new_rot = jnp.concatenate([k_rot1, k_rot2], axis=-1).astype(k_cache.dtype)
    
    def update_one(kc, vc, kn, vn, off):
        kc_out = lax.dynamic_update_slice(kc, kn, (off, 0, 0))
        vc_out = lax.dynamic_update_slice(vc, vn, (off, 0, 0))
        return kc_out, vc_out
        
    k_cache_out, v_cache_out = jax.vmap(update_one)(k_cache, v_cache, k_new_rot, v_new, offsets)
    return k_cache_out, v_cache_out
