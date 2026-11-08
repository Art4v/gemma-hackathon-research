"""Fused RMSNorm + FP8 (e4m3fn) delayed-scaling GQA QKV projection with amax tracking."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'rmsnorm_fp8_delayed_scaling_qkv',
    'batch_size': 4,
    'seq_len': 2048,
    'hidden_size': 8192,
    'num_q_heads': 64,
    'num_kv_heads': 8,
    'head_dim': 128,
    'eps': 1e-5,
    'fp8_max': 448.0,
    'margin': 0.0,
}

def create_inputs(dtype=jnp.bfloat16):
    k1, k2, k3, k4 = jax.random.split(jax.random.key(0), 4)
    B = CONFIG['batch_size']
    S = CONFIG['seq_len']
    H = CONFIG['hidden_size']
    Nq = CONFIG['num_q_heads']
    Nkv = CONFIG['num_kv_heads']
    D = CONFIG['head_dim']
    qkv_out = (Nq + 2 * Nkv) * D

    x = jax.random.normal(k1, (B, S, H), dtype=dtype)
    w_qkv = jax.random.normal(k2, (H, qkv_out), dtype=dtype) * 0.02
    rms_weight = jnp.ones((H,), dtype=dtype)
    amax_hist_x = jax.random.uniform(k3, (16,), dtype=jnp.float32, minval=1.0, maxval=4.0)
    amax_hist_w = jax.random.uniform(k4, (16,), dtype=jnp.float32, minval=0.02, maxval=0.1)
    return x, w_qkv, rms_weight, amax_hist_x, amax_hist_w

def workload(x, w_qkv, rms_weight, amax_hist_x, amax_hist_w):
    B, S, H = x.shape
    Nq = CONFIG['num_q_heads']
    Nkv = CONFIG['num_kv_heads']
    D = CONFIG['head_dim']
    eps = CONFIG['eps']
    fp8_max = CONFIG['fp8_max']
    margin = CONFIG['margin']

    # 1. RMSNorm in float32
    x_f32 = x.astype(jnp.float32)
    variance = jnp.mean(jnp.square(x_f32), axis=-1, keepdims=True)
    normed_x = x_f32 * lax.rsqrt(variance + eps) * rms_weight.astype(jnp.float32)

    # 2. Delayed scaling factors from amax history
    amax_x_prev = jnp.max(amax_hist_x)
    amax_w_prev = jnp.max(amax_hist_w)
    scale_factor = fp8_max * (2.0 ** (-margin))
    scale_x = scale_factor / jnp.where(amax_x_prev > 0.0, amax_x_prev, 1.0)
    scale_w = scale_factor / jnp.where(amax_w_prev > 0.0, amax_w_prev, 1.0)

    # 3. Compute new amax for the next iteration's delayed scaling history
    new_amax_x = jnp.max(jnp.abs(normed_x))
    w_f32 = w_qkv.astype(jnp.float32)
    new_amax_w = jnp.max(jnp.abs(w_f32))

    # 4. Quantize to FP8 e4m3fn with saturation
    x_scaled = jnp.clip(normed_x * scale_x, -fp8_max, fp8_max)
    w_scaled = jnp.clip(w_f32 * scale_w, -fp8_max, fp8_max)
    x_fp8 = x_scaled.astype(jnp.float8_e4m3fn)
    w_fp8 = w_scaled.astype(jnp.float8_e4m3fn)

    # 5. FP8 Matmul with float32 accumulation and unscaling
    x_flat = x_fp8.reshape(B * S, H)
    out_f32 = jnp.dot(x_flat, w_fp8, preferred_element_type=jnp.float32)
    inv_scale = 1.0 / (scale_x * scale_w)
    out_unscaled = (out_f32 * inv_scale).astype(x.dtype)
    out_3d = out_unscaled.reshape(B, S, (Nq + 2 * Nkv) * D)

    # 6. Split Q, K, V and reshape heads
    q_size = Nq * D
    kv_size = Nkv * D
    q = out_3d[:, :, :q_size].reshape(B, S, Nq, D)
    k = out_3d[:, :, q_size:q_size + kv_size].reshape(B, S, Nkv, D)
    v = out_3d[:, :, q_size + kv_size:].reshape(B, S, Nkv, D)

    return q, k, v, new_amax_x, new_amax_w
