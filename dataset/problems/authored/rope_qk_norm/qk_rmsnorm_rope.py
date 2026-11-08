"""Fused per-head QK-RMSNorm and Rotary Position Embedding (RoPE) for GQA."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'qk_rmsnorm_rope',
    'batch_size': 4,
    'seq_len': 4096,
    'num_q_heads': 32,
    'num_kv_heads': 8,
    'head_dim': 128,
    'eps': 1e-6,
    'rope_theta': 10000.0,
}

def create_inputs(dtype=jnp.bfloat16):
    B = CONFIG['batch_size']
    S = CONFIG['seq_len']
    Hq = CONFIG['num_q_heads']
    Hkv = CONFIG['num_kv_heads']
    D = CONFIG['head_dim']
    half_D = D // 2

    k_q, k_k = jax.random.split(jax.random.key(0), 2)
    q = jax.random.normal(k_q, (B, S, Hq, D), dtype=dtype)
    k = jax.random.normal(k_k, (B, S, Hkv, D), dtype=dtype)

    pos = jnp.arange(S, dtype=jnp.float32)[:, None]
    dim_idx = jnp.arange(half_D, dtype=jnp.float32)[None, :]
    inv_freq = 1.0 / (CONFIG['rope_theta'] ** (2.0 * dim_idx / D))
    angles = pos * inv_freq
    cos = jnp.cos(angles).astype(dtype)
    sin = jnp.sin(angles).astype(dtype)

    q_weight = jnp.ones((D,), dtype=dtype)
    k_weight = jnp.ones((D,), dtype=dtype)

    return q, k, cos, sin, q_weight, k_weight

def _rmsnorm_and_rope(x, weight, cos, sin, eps):
    orig_dtype = x.dtype
    x_f32 = x.astype(jnp.float32)
    w_f32 = weight.astype(jnp.float32)

    variance = jnp.mean(jnp.square(x_f32), axis=-1, keepdims=True)
    x_norm = x_f32 * jax.lax.rsqrt(variance + eps) * w_f32

    half_d = x.shape[-1] // 2
    x1 = x_norm[..., :half_d]
    x2 = x_norm[..., half_d:]

    cos_f32 = cos.astype(jnp.float32)[None, :, None, :]
    sin_f32 = sin.astype(jnp.float32)[None, :, None, :]

    out1 = x1 * cos_f32 - x2 * sin_f32
    out2 = x2 * cos_f32 + x1 * sin_f32
    out = jnp.concatenate([out1, out2], axis=-1)
    return out.astype(orig_dtype)

def workload(q, k, cos, sin, q_weight, k_weight):
    eps = CONFIG['eps']
    q_out = _rmsnorm_and_rope(q, q_weight, cos, sin, eps)
    k_out = _rmsnorm_and_rope(k, k_weight, cos, sin, eps)
    return q_out, k_out
