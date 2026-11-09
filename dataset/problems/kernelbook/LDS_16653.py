"""LDS_16653: three chained MaxPool2d(kernel=2, stride=2) on NCHW input."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'LDS_16653',
    'batch': 16,
    'channels': 64,
    'height': 256,
    'width': 256,
    'pool_kernel': 2,
    'pool_stride': 2,
    'num_pools': 3,
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    x = jax.random.uniform(
        key,
        (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width']),
        dtype=dtype,
    )
    return (x,)


def _maxpool2d(x):
    k = CONFIG['pool_kernel']
    s = CONFIG['pool_stride']
    return lax.reduce_window(
        x, -jnp.inf, lax.max, (1, 1, k, k), (1, 1, s, s), 'VALID'
    )


def workload(x):
    for _ in range(CONFIG['num_pools']):
        x = _maxpool2d(x)
    return x
