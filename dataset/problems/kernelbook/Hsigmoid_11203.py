"""Hardsigmoid: clip(x + 3, 0, 6) / 6."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'Hsigmoid_11203',
    'batch': 16,
    'channels': 64,
    'height': 256,
    'width': 256,
    'offset': 3.0,
    'upper': 6.0,
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    shape = (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width'])
    x = jax.random.uniform(key, shape, dtype)
    return (x,)


def workload(x):
    return jnp.clip(x + CONFIG['offset'], 0.0, CONFIG['upper']) / CONFIG['upper']
