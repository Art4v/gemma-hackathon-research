"""[bf16 variant] MaxPool2d with dynamic TF-style SAME padding (zero pad then max pool)."""
import math

import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'MaxPool2dDynamicSamePadding_2478',
    'batch': 16,
    'channels': 64,
    'height': 256,
    'width': 256,
    'kernel_size': 4,
    'stride': 1,
    'dilation': 1,
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    x = jax.random.uniform(
        key, (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width']), dtype)
    return (x,)


def _same_pad(size, k, s, d):
    out = math.ceil(size / s)
    pad = max((out - 1) * s + (k - 1) * d + 1 - size, 0)
    return (pad // 2, pad - pad // 2)


def workload(x):
    k, s, d = CONFIG['kernel_size'], CONFIG['stride'], CONFIG['dilation']
    ih, iw = x.shape[-2:]
    ph = _same_pad(ih, k, s, d)
    pw = _same_pad(iw, k, s, d)
    # F.pad uses zeros, not -inf, so padded cells take part in the max.
    xp = jnp.pad(x, ((0, 0), (0, 0), ph, pw))
    return lax.reduce_window(
        xp, -jnp.inf, lax.max,
        window_dimensions=(1, 1, k, k),
        window_strides=(1, 1, s, s),
        padding='VALID',
        window_dilation=(1, 1, d, d),
    )
