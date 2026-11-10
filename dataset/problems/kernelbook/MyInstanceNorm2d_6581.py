"""Instance norm 2d: per-instance per-channel spatial mean and population variance, then (x - mu) / sqrt(var + eps)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'MyInstanceNorm2d_6581',
    'batch': 8,
    'channels': 128,
    'height': 256,
    'width': 256,
    'eps': 1e-05,
    'spatial_axes': (2, 3),
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    shape = (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width'])
    x = jax.random.uniform(key, shape, dtype)
    return (x,)


def workload(x):
    axes = CONFIG['spatial_axes']
    mu = jnp.mean(x, axis=axes, keepdims=True)
    centered = x - mu
    sigma = jnp.mean(jnp.square(centered), axis=axes, keepdims=True)
    return centered / jnp.sqrt(sigma + CONFIG['eps'])
