"""PixelwiseNorm: x / sqrt(sum(x**2, dim=1, keepdim=True) + eps) over channels (NCHW)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'PixelwiseNorm_1129',
    'batch': 8,
    'channels': 128,
    'height': 256,
    'width': 256,
    'eps': 1e-07,
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
    return x / jnp.sqrt(jnp.sum(x ** 2, axis=1, keepdims=True) + CONFIG['eps'])
