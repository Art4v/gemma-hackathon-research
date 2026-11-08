"""QK-RMSNorm followed by rotate-half RoPE applied to GQA Q and K heads (bf16 in, f32 math)."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'qk_rmsnorm_rope_gqa',
    'batch': 2,
    'seq_len': 8192,
    'num_q_heads': 32,
    'num_kv_heads': 8,
    'head_dim': 128,
    'eps': 1e-6,
    'rope_theta': 1000000.0,
}


def create_inputs(dtype=jnp.float32):
    B = CONFIG['batch']
    S = CONFIG['seq_len']
    Hq = CONFIG['num_q_heads']
    Hk = CONFIG['num_kv_heads']
    D = CONFIG['head_dim']
    kq, kk = jax.random.split(jax.random.key(0), 2)
    q = jax.random.normal(kq, (B, S, Hq, D), dtype)
    k = jax.random.normal(kk, (B, S, Hk, D), dtype)
    positions = jnp.broadcast_to(jnp.arange(S, dtype=jnp.int32)[None, :], (B, S))
    q_scale = jnp.ones((D,), jnp.float32)
    k_scale = jnp.ones((D,), jnp.float32)
    return q, k, positions, q_scale, k_scale


def workload(q, k, positions, q_scale, k_scale):
    D = q.shape[-1]
    half = D // 2
    eps = CONFIG['eps']
    theta = CONFIG['rope_theta']

    def rmsnorm(x, scale):
        xf = x.astype(jnp.float32)
        var = jnp.mean(xf * xf, axis=-1, keepdims=True)
        return xf * lax.rsqrt(var + eps) * scale.astype(jnp.float32)

    inv_freq = 1.0 / (theta ** (jnp.arange(0, half, dtype=jnp.float32) * 2.0 / D))
    ang = positions.astype(jnp.float32)[..., None] * inv_freq  # [B, S, half]
    cos = jnp.cos(ang)[:, :, None, :]  # [B, S, 1, half]
    sin = jnp.sin(ang)[:, :, None, :]

    def rope(x):
        x1 = x[..., :half]
        x2 = x[..., half:]
        return jnp.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1)

    q_out = rope(rmsnorm(q, q_scale)).astype(q.dtype)
    k_out = rope(rmsnorm(k, k_scale)).astype(k.dtype)
    return q_out, k_out
