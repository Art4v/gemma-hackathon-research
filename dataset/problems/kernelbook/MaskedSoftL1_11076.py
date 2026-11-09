"""Masked SmoothL1 loss: (pred * mask) / factor vs target / factor, SmoothL1 (beta=1) with mean reduction."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'MaskedSoftL1_11076',
    'batch': 8,
    'channels': 64,
    'height': 256,
    'width': 256,
    'factor': 5.0,
    'beta': 1.0,
}


def create_inputs(dtype=jnp.float32):
    shape = (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width'])
    k_pred, k_target, k_mask = jax.random.split(jax.random.key(0), 3)
    pred = jax.random.uniform(k_pred, shape, dtype)
    target = jax.random.uniform(k_target, shape, dtype)
    mask = jax.random.uniform(k_mask, shape, dtype)
    return pred, target, mask


def workload(pred, target, mask):
    factor = CONFIG['factor']
    beta = CONFIG['beta']
    diff = (pred * mask) / factor - target / factor
    abs_diff = jnp.abs(diff)
    loss = jnp.where(abs_diff < beta, 0.5 * diff * diff / beta, abs_diff - 0.5 * beta)
    return jnp.mean(loss)
