"""PixelNorm: x / sqrt(mean(x**2, axis=1, keepdims=True) + eps)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'PixelNorm_18399',
    'batch': 8,
    'channels': 128,
    'height': 256,
    'width': 256,
    'eps': 1e-8,
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    x = jax.random.uniform(
        key,
        (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width']),
        dtype,
    )
    return (x,)


def workload(x):
    return x / jnp.sqrt(jnp.mean(x ** 2, axis=1, keepdims=True) + CONFIG['eps'])
