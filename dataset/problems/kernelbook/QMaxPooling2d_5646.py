"""[bf16 variant] QMaxPooling2d_5646: NCHW -> NHWC -> max_pool2d (reduce_window max) -> NCHW, with no fake quantization since qi=False."""

import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'QMaxPooling2d_5646',
    'batch': 8,
    'channels': 128,
    'height': 256,
    'width': 256,
    'kernel_size': 3,
    'stride': 1,
    'padding': 0,
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    shape = (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width'])
    x = jax.random.uniform(key, shape, dtype)
    return (x,)


def workload(x):
    k = CONFIG['kernel_size']
    s = CONFIG['stride']
    p = CONFIG['padding']
    y = jnp.transpose(x, (0, 2, 3, 1))
    y = lax.reduce_window(
        y,
        -jnp.inf,
        lax.max,
        (1, k, k, 1),
        (1, s, s, 1),
        ((0, 0), (p, p), (p, p), (0, 0)),
    )
    return jnp.transpose(y, (0, 3, 1, 2))
