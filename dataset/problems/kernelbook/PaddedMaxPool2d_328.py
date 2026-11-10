"""[bf16 variant] PaddedMaxPool2d: replicate (edge) pad followed by 2D max pooling on NCHW input."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'PaddedMaxPool2d_328',
    'batch': 8,
    'channels': 128,
    'height': 256,
    'width': 256,
    'kernel_size': 4,
    'stride': 4,
    'padding': (0, 0, 0, 0),  # (left, right, top, bottom)
    'dilation': 1,
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    shape = (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width'])
    x = jax.random.uniform(key, shape, dtype)
    return (x,)


def workload(x):
    left, right, top, bottom = CONFIG['padding']
    k = CONFIG['kernel_size']
    s = CONFIG['stride']
    d = CONFIG['dilation']
    padded = jnp.pad(x, ((0, 0), (0, 0), (top, bottom), (left, right)), mode='edge')
    return lax.reduce_window(
        padded,
        -jnp.inf,
        lax.max,
        window_dimensions=(1, 1, k, k),
        window_strides=(1, 1, s, s),
        padding='VALID',
        window_dilation=(1, 1, d, d),
    )
