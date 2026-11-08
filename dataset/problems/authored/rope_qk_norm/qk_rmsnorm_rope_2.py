"""Fused per-head QK-RMSNorm followed by Rotary Position Embedding (RoPE) for GQA queries and keys."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'qk_rmsnorm_rope',
    'batch_size': 8,
    'seq_len': 4096,
    'num_q_heads': 32,
    'num_kv_heads': 8,
    'head_dim': 128,
    'eps': 1e-6,
    'rope_theta': 10000.0,
}

def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    k_q, k_k = jax.random.split(key, 2)
    B = CONFIG['batch_size']
    S = CONFIG['seq_len']
    Hq = CONFIG['num_q_heads']
    Hkv = CONFIG['num_kv_heads']
    D = CONFIG['head_dim']

    q = jax.random.normal(k_q, (B, S, Hq, D), dtype=dtype)
    k = jax.random.normal(k_k, (B, S, Hkv, D), dtype=dtype)

    half_d = D // 2
    freq_seq = jnp.arange(0, half_d, dtype=jnp.float32) / float(half_d)
    inv_freq = 1.0 / (CONFIG['rope_theta'] ** freq_seq)
    t = jnp.arange(S, dtype=jnp.float32)
    freqs = jnp.outer(t, inv_freq)
    cos = jnp.cos(freqs).astype(dtype)
    sin = jnp.sin(freqs).astype(dtype)

    q_weight = jnp.ones((D,), dtype=dtype)
    k_weight = jnp.ones((D,), dtype=dtype)

    return q, k, cos, sin, q_weight, k_weight

def workload(q, k, cos, sin, q_weight, k_weight):
    eps = CONFIG['eps']
    D = CONFIG['head_dim']
    half_d = D // 2

    q_f32 = q.astype(jnp.float32)
    k_f32 = k.astype(jnp.float32)

    q_var = jnp.mean(jnp.square(q_f32), axis=-1, keepdims=True)
    q_norm = q_f32 * jax.lax.rsqrt(q_var + eps) * q_weight.astype(jnp.float32)

    k_var = jnp.mean(jnp.square(k_f32), axis=-1, keepdims=True)
    k_norm = k_f32 * jax.lax.rsqrt(k_var + eps) * k_weight.astype(jnp.float32)

    cos_f32 = cos.astype(jnp.float32)[None, :, None, :]
    sin_f32 = sin.astype(jnp.float32)[None, :, None, :]

    q1 = q_norm[..., :half_d]
    q2 = q_norm[..., half_d:]
    q_rot = jnp.concatenate([q1 * cos_f32 - q2 * sin_f32, q2 * cos_f32 + q1 * sin_f32], axis=-1)

    k1 = k_norm[..., :half_d]
    k2 = k_norm[..., half_d:]
    k_rot = jnp.concatenate([k1 * cos_f32 - k2 * sin_f32, k2 * cos_f32 + k1 * sin_f32], axis=-1)

    return q_rot.astype(q.dtype), k_rot.astype(k.dtype)
