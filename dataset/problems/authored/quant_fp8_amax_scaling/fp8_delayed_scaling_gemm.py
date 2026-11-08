"""FP8 E4M3FN matrix multiplication with delayed scaling and amax tracking."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'fp8_delayed_scaling_gemm',
    'M': 4096,
    'K': 8192,
    'N': 8192,
    'fp8_max': 448.0,
    'margin': 0.0,
    'eps': 1e-12,
}

def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    k1, k2 = jax.random.split(key, 2)
    M, K, N = CONFIG['M'], CONFIG['K'], CONFIG['N']
    
    x = jax.random.normal(k1, (M, K), dtype=dtype)
    w = jax.random.normal(k2, (K, N), dtype=dtype) * 0.02
    
    prev_amax_x = jnp.array([4.0], dtype=jnp.float32)
    prev_amax_w = jnp.array([0.1], dtype=jnp.float32)
    
    return x, w, prev_amax_x, prev_amax_w

def workload(x, w, prev_amax_x, prev_amax_w):
    fp8_max = CONFIG['fp8_max']
    margin = CONFIG['margin']
    eps = CONFIG['eps']
    
    eff_fp8_max = fp8_max * (2.0 ** (-margin))
    
    scale_x = eff_fp8_max / jnp.maximum(prev_amax_x, eps)
    scale_w = eff_fp8_max / jnp.maximum(prev_amax_w, eps)
    
    new_amax_x = jnp.max(jnp.abs(x)).astype(jnp.float32).reshape(1)
    new_amax_w = jnp.max(jnp.abs(w)).astype(jnp.float32).reshape(1)
    
    x_scaled = x.astype(jnp.float32) * scale_x
    x_clamped = jnp.clip(x_scaled, -fp8_max, fp8_max)
    x_fp8 = x_clamped.astype(jnp.float8_e4m3fn)
    
    w_scaled = w.astype(jnp.float32) * scale_w
    w_clamped = jnp.clip(w_scaled, -fp8_max, fp8_max)
    w_fp8 = w_clamped.astype(jnp.float8_e4m3fn)
    
    out_f32 = jnp.dot(x_fp8, w_fp8, preferred_element_type=jnp.float32)
    
    inv_scale = 1.0 / (scale_x * scale_w)
    out = (out_f32 * inv_scale).astype(x.dtype)
    
    return out, new_amax_x, new_amax_w
