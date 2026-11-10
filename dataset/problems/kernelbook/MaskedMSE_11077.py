"""Masked MSE: element-wise multiply pred by mask, then mean squared error against target."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'MaskedMSE_11077',
    'batch': 8,
    'channels': 128,
    'height': 128,
    'width': 128,
}


def create_inputs(dtype=jnp.float32):
    shape = (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width'])
    key = jax.random.key(0)
    k_pred, k_target, k_mask = jax.random.split(key, 3)
    pred = jax.random.uniform(k_pred, shape, dtype)
    target = jax.random.uniform(k_target, shape, dtype)
    mask = jax.random.uniform(k_mask, shape, dtype)
    return pred, target, mask


def workload(pred, target, mask):
    masked = jnp.multiply(pred, mask)
    return jnp.mean(jnp.square(masked - target))
