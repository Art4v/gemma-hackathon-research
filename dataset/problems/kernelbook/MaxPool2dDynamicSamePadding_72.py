"""[bf16 variant] MaxPool2dDynamicSamePadding: TF-style SAME zero padding (F.pad) followed by max_pool2d."""
import math

import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'MaxPool2dDynamicSamePadding_72',
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
        key,
        (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width']),
        dtype,
    )
    return (x,)


def _same_pad(size, k, s, d):
    out = math.ceil(size / s)
    pad = max((out - 1) * s + (k - 1) * d + 1 - size, 0)
    return (pad // 2, pad - pad // 2)


def workload(x):
    k, s, d = CONFIG['kernel_size'], CONFIG['stride'], CONFIG['dilation']
    ih, iw = x.shape[-2:]
    # F.pad pads with zeros (not -inf), so zeros take part in the max.
    x = jnp.pad(x, ((0, 0), (0, 0), _same_pad(ih, k, s, d), _same_pad(iw, k, s, d)))
    return lax.reduce_window(
        x,
        -jnp.inf,
        lax.max,
        window_dimensions=(1, 1, k, k),
        window_strides=(1, 1, s, s),
        padding='VALID',
        window_dilation=(1, 1, d, d),
    )
