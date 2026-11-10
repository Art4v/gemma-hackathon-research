"""Charbonnier loss: subtract, square, add eps, sqrt, mean."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'CharbonnierLoss_8050',
    'batch': 8,
    'channels': 128,
    'height': 128,
    'width': 256,
    'eps': 1e-06,
}


def create_inputs(dtype=jnp.float32):
    shape = (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width'])
    key_x, key_y = jax.random.split(jax.random.key(0))
    x = jax.random.uniform(key_x, shape, dtype)
    y = jax.random.uniform(key_y, shape, dtype)
    return x, y


def workload(x, y):
    diff = y - x
    error = jnp.sqrt(diff * diff + CONFIG['eps'])
    return jnp.mean(error)
