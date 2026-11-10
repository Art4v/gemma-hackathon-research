"""OutputDiscriminator: five stride-2 4x4 Conv2d layers (no bias) with LeakyReLU(0.2) between them."""

import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'OutputDiscriminator_13683',
    'batch': 8,
    'in_channels': 2,
    'height': 1024,
    'width': 1024,
    'filters': (64, 128, 256, 512, 1),
    'kernel_size': 4,
    'stride': 2,
    'padding': 2,
    'negative_slope': 0.2,
    'weight_std': 0.02,
}


def create_inputs(dtype=jnp.float32):
    filters = CONFIG['filters']
    channels = (CONFIG['in_channels'],) + filters
    k = CONFIG['kernel_size']
    keys = jax.random.split(jax.random.key(0), len(filters) + 1)
    x = jax.random.uniform(
        keys[0],
        (CONFIG['batch'], CONFIG['in_channels'], CONFIG['height'], CONFIG['width']),
        dtype,
    )
    weights = tuple(
        jax.random.normal(keys[i + 1], (channels[i + 1], channels[i], k, k), dtype)
        * CONFIG['weight_std']
        for i in range(len(filters))
    )
    return (x,) + weights


def workload(x, *weights):
    s = CONFIG['stride']
    p = CONFIG['padding']
    slope = CONFIG['negative_slope']
    y = jnp.transpose(x, (0, 2, 3, 1))
    for i, w in enumerate(weights):
        y = lax.conv_general_dilated(
            y,
            w,
            window_strides=(s, s),
            padding=((p, p), (p, p)),
            dimension_numbers=('NHWC', 'OIHW', 'NHWC'),
        )
        if i < len(weights) - 1:
            y = jnp.where(y > 0, y, slope * y)
    return jnp.transpose(y, (0, 3, 1, 2))
