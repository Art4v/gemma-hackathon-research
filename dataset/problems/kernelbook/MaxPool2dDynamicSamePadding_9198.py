"""[bf16 variant] MaxPool2d with TF-style dynamic SAME zero padding (F.pad then max_pool2d)."""
import math

import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'MaxPool2dDynamicSamePadding_9198',
    'batch': 16,
    'channels': 64,
    'height': 256,
    'width': 256,
    'kernel_size': (4, 4),
    'stride': (1, 1),
    'dilation': (1, 1),
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    x = jax.random.uniform(
        key,
        (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width']),
        dtype=dtype,
    )
    return (x,)


def _same_pad(size, k, s, d):
    out = math.ceil(size / s)
    pad = max((out - 1) * s + (k - 1) * d + 1 - size, 0)
    return (pad // 2, pad - pad // 2)


def workload(x):
    kh, kw = CONFIG['kernel_size']
    sh, sw = CONFIG['stride']
    dh, dw = CONFIG['dilation']
    ih, iw = x.shape[-2:]
    pad_h = _same_pad(ih, kh, sh, dh)
    pad_w = _same_pad(iw, kw, sw, dw)
    # F.pad pads with zeros, not -inf, so the padded cells take part in the max.
    xp = jnp.pad(x, ((0, 0), (0, 0), pad_h, pad_w))
    return lax.reduce_window(
        xp,
        -jnp.inf,
        lax.max,
        window_dimensions=(1, 1, kh, kw),
        window_strides=(1, 1, sh, sw),
        padding='VALID',
        window_dilation=(1, 1, dh, dw),
    )
