"""NasMaxPoolBlock: MaxPool2d(kernel=3, stride=2, padding=1) on NCHW input."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'NasMaxPoolBlock_17684',
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
        key,
        (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width']),
        dtype,
    )
    return (x,)


def workload(x):
    k = CONFIG['kernel_size']
    s = CONFIG['stride']
    p = CONFIG['padding']
    # PyTorch max-pool pads with -inf, so padded cells never win the max.
    return lax.reduce_window(
        x,
        jnp.array(-jnp.inf, dtype=x.dtype),
        lax.max,
        window_dimensions=(1, 1, k, k),
        window_strides=(1, 1, s, s),
        padding=((0, 0), (0, 0), (p, p), (p, p)),
    )
