"""BCNN: bilinear pooling (x x^T / HW) -> signed sqrt -> flatten -> L2 normalize."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'BCNN_15970',
    'batch': 16,
    'channels': 512,
    'height': 64,
    'width': 64,
    'thresh': 1e-8,
    'norm_eps': 1e-12,
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    x = jax.random.uniform(
        key,
        (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width']),
        dtype=dtype,
    )
    return (x,)


def workload(x):
    b, c, h, w = x.shape
    xf = x.reshape(b, c, h * w)
    gram = jnp.einsum('bcn,bdn->bcd', xf, xf, precision=lax.Precision.HIGHEST)
    gram = gram * (1.0 / (h * w))
    y = jnp.sign(gram) * jnp.sqrt(jnp.abs(gram) + CONFIG['thresh'])
    y = y.reshape(b, -1)
    norm = jnp.sqrt(jnp.sum(y * y, axis=1, keepdims=True))
    return y / jnp.maximum(norm, CONFIG['norm_eps'])
