"""[bf16 variant] ZeroPad2d (left/top) -> MaxPool2d (kernel 3, stride 2, padding 1) -> crop first row and column, NCHW."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'MaxPoolPad_9868',
    'batch': 8,
    'channels': 128,
    'height': 256,
    'width': 256,
    'pad_left': 1,
    'pad_right': 0,
    'pad_top': 1,
    'pad_bottom': 0,
    'pool_kernel': 3,
    'pool_stride': 2,
    'pool_padding': 1,
    'crop': 1,
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    shape = (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width'])
    x = jax.random.uniform(key, shape, dtype)
    return (x,)


def workload(x):
    x = jnp.pad(
        x,
        (
            (0, 0),
            (0, 0),
            (CONFIG['pad_top'], CONFIG['pad_bottom']),
            (CONFIG['pad_left'], CONFIG['pad_right']),
        ),
        mode='constant',
        constant_values=0,
    )
    kernel = CONFIG['pool_kernel']
    stride = CONFIG['pool_stride']
    padding = CONFIG['pool_padding']
    x = lax.reduce_window(
        x,
        jnp.array(-jnp.inf, x.dtype),
        lax.max,
        window_dimensions=(1, 1, kernel, kernel),
        window_strides=(1, 1, stride, stride),
        padding=((0, 0), (0, 0), (padding, padding), (padding, padding)),
    )
    crop = CONFIG['crop']
    return x[:, :, crop:, crop:]
