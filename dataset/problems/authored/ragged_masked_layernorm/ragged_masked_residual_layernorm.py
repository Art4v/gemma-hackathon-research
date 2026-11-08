"""Fused residual-add + LayerNorm over right-padded ragged sequences with padded positions masked to zero."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'ragged_masked_residual_layernorm',
    'batch': 8,
    'seq_len': 4096,
    'hidden': 2048,
    'eps': 1e-5,
    'min_len': 1,
}


def create_inputs(dtype=jnp.bfloat16):
    B, S, D = CONFIG['batch'], CONFIG['seq_len'], CONFIG['hidden']
    key = jax.random.key(0)
    kx, kr, kl = jax.random.split(key, 3)
    x = jax.random.normal(kx, (B, S, D), dtype=dtype)
    residual = jax.random.normal(kr, (B, S, D), dtype=dtype)
    gamma = jnp.ones((D,), dtype=dtype)
    beta = jnp.zeros((D,), dtype=dtype)
    lengths = jax.random.randint(kl, (B,), CONFIG['min_len'], S + 1, dtype=jnp.int32)
    return x, residual, gamma, beta, lengths


def workload(x, residual, gamma, beta, lengths):
    B, S, D = x.shape
    eps = CONFIG['eps']
    pos = jnp.arange(S, dtype=jnp.int32)
    mask = (pos[None, :] < lengths[:, None])[:, :, None]  # [B, S, 1]

    h = x.astype(jnp.float32) + residual.astype(jnp.float32)
    h = jnp.where(mask, h, 0.0)

    mean = jnp.mean(h, axis=-1, keepdims=True)
    centered = h - mean
    var = jnp.mean(centered * centered, axis=-1, keepdims=True)
    inv = jax.lax.rsqrt(var + eps)
    y = centered * inv * gamma.astype(jnp.float32) + beta.astype(jnp.float32)
    y = jnp.where(mask, y, 0.0)

    return y.astype(x.dtype), h.astype(x.dtype)
