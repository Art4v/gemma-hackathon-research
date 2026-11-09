"""Multi-scale MaxPool2d (k2/s2/p0, k4/s2/p1, k8/s2/p3) averaged: (x1 + x2 + x3) / 3."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'multi_pool_6124',
    'batch': 16,
    'channels': 64,
    'height': 256,
    'width': 256,
    'stride': 2,
    'pools': ((2, 0), (4, 1), (8, 3)),  # (kernel, padding)
    'divisor': 3.0,
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    x = jax.random.uniform(
        key, (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width']), dtype)
    return (x,)


def _max_pool(x, k, s, p):
    # PyTorch MaxPool2d pads with -inf, so padded cells never win the max.
    return lax.reduce_window(
        x, -jnp.inf, lax.max,
        window_dimensions=(1, 1, k, k),
        window_strides=(1, 1, s, s),
        padding=((0, 0), (0, 0), (p, p), (p, p)),
    )


def workload(x):
    s = CONFIG['stride']
    pooled = [_max_pool(x, k, s, p) for k, p in CONFIG['pools']]
    return (pooled[0] + pooled[1] + pooled[2]) / CONFIG['divisor']
