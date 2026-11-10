"""[bf16 variant] ZeroPad2d (left/top) -> AvgPool2d (count_include_pad=False) -> crop first row/column, on NCHW input."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'AvgPoolPad_2297',
    'batch': 8,
    'channels': 128,
    'height': 256,
    'width': 256,
    'pad_left': 1,
    'pad_right': 0,
    'pad_top': 1,
    'pad_bottom': 0,
    'kernel_size': 3,
    'stride': 2,
    'padding': 1,
    'crop': 1,
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    shape = (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width'])
    x = jax.random.uniform(key, shape, dtype)
    return (x,)


def workload(x):
    padded = jnp.pad(
        x,
        (
            (0, 0),
            (0, 0),
            (CONFIG['pad_top'], CONFIG['pad_bottom']),
            (CONFIG['pad_left'], CONFIG['pad_right']),
        ),
    )

    k = CONFIG['kernel_size']
    s = CONFIG['stride']
    p = CONFIG['padding']
    window = (1, 1, k, k)
    strides = (1, 1, s, s)
    pool_padding = ((0, 0), (0, 0), (p, p), (p, p))

    summed = lax.reduce_window(padded, 0.0, lax.add, window, strides, pool_padding)
    ones = jnp.ones((1, 1) + padded.shape[2:], padded.dtype)
    counts = lax.reduce_window(ones, 0.0, lax.add, window, strides, pool_padding)
    pooled = summed / counts

    c = CONFIG['crop']
    return pooled[:, :, c:, c:]
