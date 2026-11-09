"""FusedUpsample: equalized-lr weight scale -> 2x2 kernel blur -> conv_transpose2d(stride=2) + bias."""
import math

import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'FusedUpsample_16645',
    'batch': 8,
    'in_channel': 128,
    'out_channel': 128,
    'height': 128,
    'width': 128,
    'kernel_size': 4,
    'stride': 2,
    'padding': 0,
}


def create_inputs(dtype=jnp.float32):
    kx, kw = jax.random.split(jax.random.key(0))
    c = CONFIG
    x = jax.random.uniform(kx, (c['batch'], c['in_channel'], c['height'], c['width']), dtype)
    weight = jax.random.normal(
        kw, (c['in_channel'], c['out_channel'], c['kernel_size'], c['kernel_size']), dtype) * 0.02
    bias = jnp.zeros((c['out_channel'],), dtype)
    return x, weight, bias


def workload(x, weight, bias):
    cin, _, k, _ = weight.shape
    stride = CONFIG['stride']
    pad = CONFIG['padding']
    multiplier = math.sqrt(2.0 / (cin * k * k))
    w = jnp.pad(weight * multiplier, ((0, 0), (0, 0), (1, 1), (1, 1)))
    w = (w[:, :, 1:, 1:] + w[:, :, :-1, 1:] + w[:, :, 1:, :-1] + w[:, :, :-1, :-1]) / 4
    kb = w.shape[-1]
    # conv_transpose == dilated-input conv with spatially flipped kernel, (in,out,H,W) -> HWIO
    kernel = jnp.transpose(w[:, :, ::-1, ::-1], (2, 3, 0, 1))
    x_nhwc = jnp.transpose(x, (0, 2, 3, 1))
    edge = kb - 1 - pad
    out = lax.conv_general_dilated(
        x_nhwc, kernel,
        window_strides=(1, 1),
        padding=((edge, edge), (edge, edge)),
        lhs_dilation=(stride, stride),
        dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
        precision=lax.Precision.HIGHEST,
    )
    out = out + bias
    return jnp.transpose(out, (0, 3, 1, 2))
