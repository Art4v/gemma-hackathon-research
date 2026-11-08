"""Fused per-token per-head int8 KV quantization and cache scatter write."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'kv_cache_int8_write',
    'batch_size': 32,
    'max_seq_len': 4096,
    'update_len': 128,
    'num_kv_heads': 8,
    'head_dim': 128,
    'quant_max': 127.0,
    'eps': 1e-6,
}

def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    k1, k2, k3, k4, k5, k6, k7 = jax.random.split(key, 7)
    
    B = CONFIG['batch_size']
    S = CONFIG['max_seq_len']
    U = CONFIG['update_len']
    H = CONFIG['num_kv_heads']
    D = CONFIG['head_dim']
    
    new_k = jax.random.normal(k1, (B, U, H, D), dtype=dtype)
    new_v = jax.random.normal(k2, (B, U, H, D), dtype=dtype)
    
    k_cache = jax.random.randint(k3, (B, S, H, D), minval=-127, maxval=128, dtype=jnp.int8)
    v_cache = jax.random.randint(k4, (B, S, H, D), minval=-127, maxval=128, dtype=jnp.int8)
    
    k_scale_cache = jax.random.uniform(k5, (B, S, H), dtype=jnp.float32, minval=0.001, maxval=0.05)
    v_scale_cache = jax.random.uniform(k6, (B, S, H), dtype=jnp.float32, minval=0.001, maxval=0.05)
    
    start_positions = jax.random.randint(k7, (B,), minval=0, maxval=S - U + 1, dtype=jnp.int32)
    
    return new_k, new_v, k_cache, v_cache, k_scale_cache, v_scale_cache, start_positions

def workload(new_k, new_v, k_cache, v_cache, k_scale_cache, v_scale_cache, start_positions):
    qmax = CONFIG['quant_max']
    eps = CONFIG['eps']
    
    new_k_f32 = new_k.astype(jnp.float32)
    new_v_f32 = new_v.astype(jnp.float32)
    
    k_absmax = jnp.max(jnp.abs(new_k_f32), axis=-1)
    v_absmax = jnp.max(jnp.abs(new_v_f32), axis=-1)
    
    k_scale = jnp.maximum(k_absmax, eps) / qmax
    v_scale = jnp.maximum(v_absmax, eps) / qmax
    
    k_quant = jnp.clip(jnp.round(new_k_f32 / k_scale[..., None]), -qmax, qmax).astype(jnp.int8)
    v_quant = jnp.clip(jnp.round(new_v_f32 / v_scale[..., None]), -qmax, qmax).astype(jnp.int8)
    
    def update_single(kc, vc, ksc, vsc, kq, vq, ks, vs, pos):
        kc_out = lax.dynamic_update_slice(kc, kq, (pos, 0, 0))
        vc_out = lax.dynamic_update_slice(vc, vq, (pos, 0, 0))
        ksc_out = lax.dynamic_update_slice(ksc, ks, (pos, 0))
        vsc_out = lax.dynamic_update_slice(vsc, vs, (pos, 0))
        return kc_out, vc_out, ksc_out, vsc_out
        
    return jax.vmap(update_single)(
        k_cache, v_cache, k_scale_cache, v_scale_cache,
        k_quant, v_quant, k_scale, v_scale, start_positions
    )
