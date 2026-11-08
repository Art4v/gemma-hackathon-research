"""Fused RMSNorm and FP8 (e4m3fn) delayed-scaling QKV projection with amax history update."""

import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'rmsnorm_fp8_delayed_scaling_qkv',
    'batch_size': 4,
    'seq_len': 2048,
    'd_model': 8192,
    'num_q_heads': 64,
    'num_kv_heads': 8,
    'head_dim': 128,
    'amax_history_len': 16,
    'eps': 1e-5,
    'fp8_max': 448.0,
    'margin': 0.0,
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    k1, k2, k3, k4 = jax.random.split(key, 4)
    B = CONFIG['batch_size']
    S = CONFIG['seq_len']
    D = CONFIG['d_model']
    H_q = CONFIG['num_q_heads']
    H_kv = CONFIG['num_kv_heads']
    D_h = CONFIG['head_dim']
    H_len = CONFIG['amax_history_len']
    total_heads = H_q + 2 * H_kv
    out_dim = total_heads * D_h

    x = jax.random.normal(k1, (B, S, D), dtype=dtype)
    norm_weight = jnp.ones((D,), dtype=dtype)
    w_qkv = jax.random.normal(k2, (D, out_dim), dtype=dtype) * 0.02
    b_qkv = jnp.zeros((out_dim,), dtype=dtype)

    amax_history_x = jax.random.uniform(k3, (H_len,), dtype=jnp.float32, minval=0.5, maxval=4.0)
    amax_history_w = jax.random.uniform(k4, (H_len,), dtype=jnp.float32, minval=0.01, maxval=0.1)

    return x, norm_weight, w_qkv, b_qkv, amax_history_x, amax_history_w


def workload(x, norm_weight, w_qkv, b_qkv, amax_history_x, amax_history_w):
    B, S, D = x.shape
    H_q = CONFIG['num_q_heads']
    H_kv = CONFIG['num_kv_heads']
    D_h = CONFIG['head_dim']
    eps = CONFIG['eps']
    fp8_max = CONFIG['fp8_max']
    margin = CONFIG['margin']
    orig_dtype = x.dtype

    # 1. Pre-Attention RMSNorm in float32
    x_f32 = x.astype(jnp.float32)
    variance = jnp.mean(jnp.square(x_f32), axis=-1, keepdims=True)
    x_norm = x_f32 * lax.rsqrt(variance + eps) * norm_weight.astype(jnp.float32)

    w_f32 = w_qkv.astype(jnp.float32)

    # 2. Compute current amax from normalized activations and weights
    curr_amax_x = jnp.max(jnp.abs(x_norm))
    curr_amax_w = jnp.max(jnp.abs(w_f32))

    # 3. Derive delayed scaling factors from prior amax history
    hist_max_x = jnp.max(amax_history_x)
    hist_max_w = jnp.max(amax_history_w)
    sf_factor = fp8_max / (2.0 ** margin)
    scale_x = jnp.where(hist_max_x > 0.0, sf_factor / hist_max_x, 1.0)
    scale_w = jnp.where(hist_max_w > 0.0, sf_factor / hist_max_w, 1.0)

    # 4. Roll amax history buffers and record current amax
    new_amax_history_x = jnp.roll(amax_history_x, -1).at[-1].set(curr_amax_x)
    new_amax_history_w = jnp.roll(amax_history_w, -1).at[-1].set(curr_amax_w)

    # 5. Saturate and cast to FP8 E4M3
    x_scaled = jnp.clip(x_norm * scale_x, -fp8_max, fp8_max)
    w_scaled = jnp.clip(w_f32 * scale_w, -fp8_max, fp8_max)
    x_fp8 = x_scaled.astype(jnp.float8_e4m3fn)
    w_fp8 = w_scaled.astype(jnp.float8_e4m3fn)

    # 6. FP8 Matmul with float32 accumulation
    x_2d = x_fp8.reshape(B * S, D)
    out_f32 = jnp.dot(x_2d, w_fp8, preferred_element_type=jnp.float32)

    # 7. Unscale, add bias, and compute output amax
    inv_scale = 1.0 / (scale_x * scale_w)
    out_unscaled = out_f32 * inv_scale + b_qkv.astype(jnp.float32)
    out_amax = jnp.max(jnp.abs(out_unscaled))

    # 8. Split and reshape into GQA Query, Key, Value tensors
    out_3d = out_unscaled.astype(orig_dtype).reshape(B, S, H_q + 2 * H_kv, D_h)
    q = out_3d[:, :, :H_q, :]
    k = out_3d[:, :, H_q:H_q + H_kv, :]
    v = out_3d[:, :, H_q + H_kv:, :]

    return q, k, v, new_amax_history_x, new_amax_history_w, out_amax
