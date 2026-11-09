"""[bf16 variant] MaxPool2d (kernel 4, stride 1, padding 1, dilation 1) over an NCHW tensor."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': '42_Max_Pooling_2D',
    'batch_size': 16,
    'channels': 64,
    'height': 256,
    'width': 256,
    'kernel_size': 4,
    'stride': 1,
    'padding': 1,
    'dilation': 1,
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
    s = CONFIG['stride']
    p = CONFIG['padding']
    d = CONFIG['dilation']
    init = jnp.array(-jnp.inf, dtype=x.dtype)
    return lax.reduce_window(
        x,
        init,
        lax.max,
        window_dimensions=(1, 1, k, k),
        window_strides=(1, 1, s, s),
        padding=((0, 0), (0, 0), (p, p), (p, p)),
        window_dilation=(1, 1, d, d),
    )
