"""[bf16 variant] MaxPool2d (kernel 4, stride 1, padding 1) on NCHW input."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'MaxPool_17153',
    'batch': 16,
    'channels': 64,
    'height': 256,
    'width': 256,
    'kernel_size': 4,
    'stride': 1,
    'padding': 1,
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    x = jax.random.uniform(
        key,
        (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width']),
        dtype=dtype,
    )
    return (x,)


def workload(x):
    k = CONFIG['kernel_size']
    s = CONFIG['stride']
    p = CONFIG['padding']
    # PyTorch MaxPool2d pads with -inf, so padded cells never win the max.
    return lax.reduce_window(
        x,
        jnp.array(-jnp.inf, dtype=x.dtype),
        lax.max,
        window_dimensions=(1, 1, k, k),
        window_strides=(1, 1, s, s),
        padding=((0, 0), (0, 0), (p, p), (p, p)),
    )
