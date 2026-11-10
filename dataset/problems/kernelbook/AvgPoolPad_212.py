"""ZeroPad2d (left/top) -> AvgPool2d (count_include_pad=False) -> crop first row and column."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'AvgPoolPad_212',
    'batch': 8,
    'channels': 128,
    'height': 256,
    'width': 256,
    'zero_pad': (1, 0, 1, 0),  # (left, right, top, bottom)
    'pool_window': 3,
    'pool_stride': 2,
    'pool_padding': 1,
    'crop': 1,
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    shape = (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width'])
    x = jax.random.uniform(key, shape, dtype)
    return (x,)


def workload(x):
    left, right, top, bottom = CONFIG['zero_pad']
    window = CONFIG['pool_window']
    stride = CONFIG['pool_stride']
    pool_pad = CONFIG['pool_padding']
    crop = CONFIG['crop']

    x = jnp.pad(x, ((0, 0), (0, 0), (top, bottom), (left, right)))

    window_dims = (1, 1, window, window)
    strides = (1, 1, stride, stride)
    padding = ((0, 0), (0, 0), (pool_pad, pool_pad), (pool_pad, pool_pad))

    summed = lax.reduce_window(x, 0.0, lax.add, window_dims, strides, padding)
    ones = jnp.ones((1, 1, x.shape[2], x.shape[3]), x.dtype)
    counts = lax.reduce_window(ones, 0.0, lax.add, window_dims, strides, padding)
    out = summed / counts

    return out[:, :, crop:, crop:]
