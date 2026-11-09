"""ConstantPad1d (zero, same-style) followed by MaxPool1d over the last axis."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'SampaddingMaxPool1D_6195',
    'channels': 1024,
    'length': 65536,
    'pooling_size': 4,
    'stride': 1,
    'pad_value': 0.0,
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    x = jax.random.uniform(key, (CONFIG['channels'], CONFIG['length']), dtype)
    return (x,)


def workload(x):
    p = CONFIG['pooling_size']
    s = CONFIG['stride']
    pad_left = (p - 1) // 2
    pad_right = p // 2
    pad_width = [(0, 0)] * (x.ndim - 1) + [(pad_left, pad_right)]
    # Zero padding (not -inf): the padded zeros take part in the max, as in PyTorch.
    xp = jnp.pad(x, pad_width, mode='constant',
                 constant_values=jnp.asarray(CONFIG['pad_value'], x.dtype))
    window = (1,) * (x.ndim - 1) + (p,)
    strides = (1,) * (x.ndim - 1) + (s,)
    return lax.reduce_window(xp, jnp.array(-jnp.inf, x.dtype), lax.max,
                             window, strides, 'VALID')
