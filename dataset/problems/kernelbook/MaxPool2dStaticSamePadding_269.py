"""[bf16 variant] MaxPool2dStaticSamePadding: TF-style same zero-padding (F.pad) followed by MaxPool2d(kernel=4, stride=4)."""
import math

import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'MaxPool2dStaticSamePadding_269',
    'batch': 16,
    'channels': 64,
    'height': 256,
    'width': 256,
    'kernel_size': 4,
    'stride': 4,  # nn.MaxPool2d defaults stride to kernel_size
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
    h, w = x.shape[-2:]
    extra_h = (math.ceil(w / s) - 1) * s - w + k
    extra_v = (math.ceil(h / s) - 1) * s - h + k
    left = extra_h // 2
    right = extra_h - left
    top = extra_v // 2
    bottom = extra_v - top
    # F.pad pads with constant 0, not -inf, so padded cells take part in the max.
    x = jnp.pad(x, ((0, 0), (0, 0), (top, bottom), (left, right)))
    return lax.reduce_window(
        x,
        -jnp.inf,
        lax.max,
        window_dimensions=(1, 1, k, k),
        window_strides=(1, 1, s, s),
        padding='VALID',
    )
