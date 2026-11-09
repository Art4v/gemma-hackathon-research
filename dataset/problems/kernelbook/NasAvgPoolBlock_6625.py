"""NasAvgPoolBlock: AvgPool2d(3, stride=2, padding=1, count_include_pad=False) (extra_padding=False)."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'NasAvgPoolBlock_6625',
    'batch': 8,
    'channels': 128,
    'height': 256,
    'width': 256,
    'kernel_size': 3,
    'stride': 2,
    'padding': 1,
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    x = jax.random.uniform(
        key, (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width']), dtype)
    return (x,)


def workload(x):
    k, s, p = CONFIG['kernel_size'], CONFIG['stride'], CONFIG['padding']
    window = (1, 1, k, k)
    strides = (1, 1, s, s)
    pads = ((0, 0), (0, 0), (p, p), (p, p))
    sums = lax.reduce_window(x, jnp.array(0, x.dtype), lax.add, window, strides, pads)
    # count_include_pad=False: divide by the number of real (non-padded) elements per window.
    ones = jnp.ones((1, 1) + x.shape[2:], x.dtype)
    counts = lax.reduce_window(ones, jnp.array(0, x.dtype), lax.add, window, strides, pads)
    return sums / counts
