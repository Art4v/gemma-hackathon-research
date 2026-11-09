"""Noise (eval): scale by 255, round, clamp to [0, 255], divide by 255."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'Noise_15349',
    'batch': 16,
    'channels': 64,
    'height': 256,
    'width': 256,
    'scale': 255.0,
}


def create_inputs(dtype=jnp.float32):
    shape = (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width'])
    x = jax.random.uniform(jax.random.key(0), shape, dtype)
    return (x,)


def workload(x):
    scale = CONFIG['scale']
    y = jnp.round(x * scale)
    y = jnp.clip(y, 0.0, scale)
    return y / scale
