"""[bf16 variant] 2D Average Pooling."""

import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': '45_Average_Pooling_2D',
    'batch_size': 8,
    'channels': 32,
    'height': 512,
    'width': 512,
    'kernel_size': 11,
    'stride': 11,
    'padding': 0,
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    x = jax.random.uniform(
        key,
        (CONFIG['batch_size'], CONFIG['channels'], CONFIG['height'], CONFIG['width']),
        dtype=dtype,
    )
    return (x,)


def workload(x):
    k = CONFIG['kernel_size']
    s = CONFIG['stride'] if CONFIG['stride'] is not None else k
    p = CONFIG['padding']
    window_shape = (1, 1, k, k)
    strides = (1, 1, s, s)
    padding = ((0, 0), (0, 0), (p, p), (p, p))
    pooled = lax.reduce_window(x, 0.0, lax.add, window_shape, strides, padding)
    return pooled / (k * k)
