"""Ragged masked fused residual-add + LayerNorm over padded variable-length sequences (padded positions zeroed)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'ragged_masked_add_layernorm',
    'batch': 16,
    'seq_len': 4096,
    'hidden': 2048,
    'eps': 1e-5,
    'min_len': 1,
}


def create_inputs(dtype=jnp.bfloat16):
    B, S, D = CONFIG['batch'], CONFIG['seq_len'], CONFIG['hidden']
    key = jax.random.key(0)
    k_x, k_r, k_len = jax.random.split(key, 3)
    x = jax.random.normal(k_x, (B, S, D), dtype=dtype)
    residual = jax.random.normal(k_r, (B, S, D), dtype=dtype)
    gamma = jnp.ones((D,), dtype=dtype)
    beta = jnp.zeros((D,), dtype=dtype)
    lengths = jax.random.randint(k_len, (B,), CONFIG['min_len'], S + 1, dtype=jnp.int32)
    return x, residual, gamma, beta, lengths


def workload(x, residual, gamma, beta, lengths):
    eps = CONFIG['eps']
    S = x.shape[1]
    out_dtype = x.dtype

    # Valid-token mask: [B, S, 1]
    pos = jnp.arange(S, dtype=jnp.int32)
    mask = (pos[None, :] < lengths[:, None])[..., None]

    # Fused residual add in float32.
    h = x.astype(jnp.float32) + residual.astype(jnp.float32)
    # Zero padded positions before statistics so garbage padding never propagates.
    h = jnp.where(mask, h, 0.0)

    mean = jnp.mean(h, axis=-1, keepdims=True)
    centered = h - mean
    var = jnp.mean(centered * centered, axis=-1, keepdims=True)
    inv = jax.lax.rsqrt(var + eps)
    y = centered * inv * gamma.astype(jnp.float32) + beta.astype(jnp.float32)

    y = jnp.where(mask, y, 0.0)
    new_residual = h  # already zero at padded positions

    return y.astype(out_dtype), new_residual.astype(out_dtype)
