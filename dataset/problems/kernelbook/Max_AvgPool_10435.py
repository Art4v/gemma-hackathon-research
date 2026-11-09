"""MaxPool2d(3x3, s2, p1) + AvgPool2d(3x3, s2, p1, count_include_pad=True), summed."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'Max_AvgPool_10435',
    'batch': 16,
    'channels': 64,
    'height': 256,
    'width': 256,
    'kernel_size': (3, 3),
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
    kh, kw = CONFIG['kernel_size']
    s = CONFIG['stride']
    p = CONFIG['padding']
    window = (1, 1, kh, kw)
    strides = (1, 1, s, s)
    pads = ((0, 0), (0, 0), (p, p), (p, p))
    mx = lax.reduce_window(x, jnp.array(-jnp.inf, x.dtype), lax.max, window, strides, pads)
    # count_include_pad=True (PyTorch default): zero padding, divide by full window size.
    sm = lax.reduce_window(x, jnp.array(0, x.dtype), lax.add, window, strides, pads)
    return mx + sm / (kh * kw)
