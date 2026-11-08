"""Ragged masked LayerNorm: residual add -> LayerNorm (fp32 stats) -> affine -> zero padded positions beyond per-sequence length."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'ragged_masked_layernorm',
    'batch': 8,
    'seq_len': 4096,
    'hidden': 2048,
    'eps': 1e-5,
    'min_len': 1,
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    k_x, k_r, k_len = jax.random.split(key, 3)
    B, S, D = CONFIG['batch'], CONFIG['seq_len'], CONFIG['hidden']
    x = jax.random.normal(k_x, (B, S, D), dtype)
    residual = jax.random.normal(k_r, (B, S, D), dtype)
    lengths = jax.random.randint(k_len, (B,), CONFIG['min_len'], S + 1, dtype=jnp.int32)
    gamma = jnp.ones((D,), dtype)
    beta = jnp.zeros((D,), dtype)
    return x, residual, lengths, gamma, beta


def workload(x, residual, lengths, gamma, beta):
    eps = CONFIG['eps']
    S = x.shape[1]
    out_dtype = x.dtype
    # Validity mask from per-sequence lengths: [B, S, 1]
    pos = jnp.arange(S, dtype=jnp.int32)[None, :]
    mask = (pos < lengths[:, None])[..., None]

    h = x.astype(jnp.float32) + residual.astype(jnp.float32)
    mean = jnp.mean(h, axis=-1, keepdims=True)
    centered = h - mean
    var = jnp.mean(centered * centered, axis=-1, keepdims=True)
    normed = centered * jax.lax.rsqrt(var + eps)
    normed = normed * gamma.astype(jnp.float32) + beta.astype(jnp.float32)

    normed = jnp.where(mask, normed, 0.0).astype(out_dtype)
    new_residual = jnp.where(mask, h, 0.0).astype(out_dtype)
    return normed, new_residual
