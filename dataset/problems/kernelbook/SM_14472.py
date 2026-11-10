"""AvgPool2d(k=3, s=1, p=1) -> MaxPool2d(k=3, s=1, p=1) on NCHW input."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'SM_14472',
    'batch': 8,
    'channels': 128,
    'height': 256,
    'width': 256,
    'kernel_size': 3,
    'stride': 1,
    'padding': 1,
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    shape = (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width'])
    x = jax.random.uniform(key, shape, dtype)
    return (x,)


def workload(x):
    k = CONFIG['kernel_size']
    s = CONFIG['stride']
    p = CONFIG['padding']
    window = (1, 1, k, k)
    strides = (1, 1, s, s)
    padding = ((0, 0), (0, 0), (p, p), (p, p))
    # count_include_pad=True: zeros in the padded region count, divisor is always k*k.
    avg = lax.reduce_window(x, 0.0, lax.add, window, strides, padding) / (k * k)
    # -inf padding keeps padded cells from contributing to the max.
    return lax.reduce_window(avg, -jnp.inf, lax.max, window, strides, padding)
