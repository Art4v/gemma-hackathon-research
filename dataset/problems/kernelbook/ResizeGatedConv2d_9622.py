"""[bf16 variant] Nearest upsample x2 -> gated Conv2d (conv_h(x) * sigmoid(conv_g(x)))."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'ResizeGatedConv2d_9622',
    'batch': 8,
    'in_channels': 128,
    'out_channels': 128,
    'height': 128,
    'width': 128,
    'kernel_size': 4,
    'stride': 1,
    'padding': 4,
    'dilation': 1,
    'scale_factor': 2,
}


def create_inputs(dtype=jnp.bfloat16):
    c = CONFIG
    k_x, k_wh, k_bh, k_wg, k_bg = jax.random.split(jax.random.key(0), 5)
    x = jax.random.uniform(
        k_x, (c['batch'], c['in_channels'], c['height'], c['width']), dtype)
    w_shape = (c['kernel_size'], c['kernel_size'], c['in_channels'], c['out_channels'])
    w_h = jax.random.normal(k_wh, w_shape, dtype) * 0.02
    b_h = jax.random.normal(k_bh, (c['out_channels'],), dtype) * 0.02
    w_g = jax.random.normal(k_wg, w_shape, dtype) * 0.02
    b_g = jax.random.normal(k_bg, (c['out_channels'],), dtype) * 0.02
    return x, w_h, b_h, w_g, b_g


def _conv(x_nhwc, w_hwio, b):
    p = CONFIG['padding']
    s = CONFIG['stride']
    d = CONFIG['dilation']
    y = lax.conv_general_dilated(
        x_nhwc, w_hwio,
        window_strides=(s, s),
        padding=((p, p), (p, p)),
        rhs_dilation=(d, d),
        dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
        precision=lax.Precision.HIGHEST,
    )
    return y + b


def workload(x, w_h, b_h, w_g, b_g):
    sf = CONFIG['scale_factor']
    x = jnp.transpose(x, (0, 2, 3, 1))
    x = jnp.repeat(jnp.repeat(x, sf, axis=1), sf, axis=2)
    out = _conv(x, w_h, b_h) * jax.nn.sigmoid(_conv(x, w_g, b_g))
    return jnp.transpose(out, (0, 3, 1, 2))
