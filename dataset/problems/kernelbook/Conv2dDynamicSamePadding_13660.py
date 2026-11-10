"""[bf16 variant] Conv2dDynamicSamePadding: TF-style SAME pad (asymmetric) -> Conv2d + bias."""
import math

import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'Conv2dDynamicSamePadding_13660',
    'batch': 8,
    'in_channels': 128,
    'out_channels': 128,
    'height': 256,
    'width': 256,
    'kernel_size': 4,
    'stride': 1,
    'dilation': 1,
    'groups': 1,
}


def create_inputs(dtype=jnp.bfloat16):
    c = CONFIG
    kx, kw, kb = jax.random.split(jax.random.key(0), 3)
    x = jax.random.uniform(kx, (c['batch'], c['in_channels'], c['height'], c['width']), dtype)
    w = jax.random.normal(
        kw, (c['out_channels'], c['in_channels'] // c['groups'], c['kernel_size'], c['kernel_size']), dtype
    ) * 0.02
    b = jax.random.normal(kb, (c['out_channels'],), dtype) * 0.02
    return x, w, b


def _same_pad(size, k, s, d):
    out = math.ceil(size / s)
    total = max((out - 1) * s + (k - 1) * d + 1 - size, 0)
    return (total // 2, total - total // 2)


def workload(x, w, b):
    s, d, g = CONFIG['stride'], CONFIG['dilation'], CONFIG['groups']
    ih, iw = x.shape[-2:]
    kh, kw = w.shape[-2:]
    padding = (_same_pad(ih, kh, s, d), _same_pad(iw, kw, s, d))
    x_nhwc = jnp.transpose(x, (0, 2, 3, 1))
    w_hwio = jnp.transpose(w, (2, 3, 1, 0))
    y = lax.conv_general_dilated(
        x_nhwc, w_hwio,
        window_strides=(s, s),
        padding=padding,
        rhs_dilation=(d, d),
        dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
        feature_group_count=g,
    )
    y = y + b
    return jnp.transpose(y, (0, 3, 1, 2))
