"""Fused pre-norm residual block: residual add + Gemma-style RMSNorm returning (normed, new_residual)."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'fused_add_rmsnorm_residual',
    'batch': 4,
    'seq_len': 4096,
    'hidden': 5376,
    'eps': 1e-6,
}


def create_inputs(dtype=jnp.float32):
    b = CONFIG['batch']
    s = CONFIG['seq_len']
    h = CONFIG['hidden']
    key = jax.random.key(0)
    k_x, k_res = jax.random.split(key, 2)
    # Sub-layer output (attention / MLP branch) entering the residual add.
    x = jax.random.normal(k_x, (b, s, h), dtype)
    # Incoming residual stream.
    residual = jax.random.normal(k_res, (b, s, h), dtype)
    # Gemma-style zero-centred RMSNorm scale, kept in f32 like the model does.
    scale = jnp.zeros((h,), jnp.float32)
    return (x, residual, scale)


def workload(x, residual, scale):
    eps = CONFIG['eps']
    # Residual add in f32; the sum is the updated residual stream (stored in activation dtype).
    h = residual.astype(jnp.float32) + x.astype(jnp.float32)
    new_residual = h.astype(x.dtype)
    # RMSNorm with f32 statistics over the hidden axis.
    var = jnp.mean(jnp.square(h), axis=-1, keepdims=True)
    normed = h * lax.rsqrt(var + eps)
    normed = normed * (1.0 + scale.astype(jnp.float32))
    return normed.astype(x.dtype), new_residual
