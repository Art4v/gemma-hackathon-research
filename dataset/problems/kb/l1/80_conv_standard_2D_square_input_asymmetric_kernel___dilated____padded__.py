"""Conv2d (asymmetric kernel 5x9, dilation (2,3), padding (2,4), no bias)."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': '80_conv_standard_2D_square_input_asymmetric_kernel___dilated____padded__',
    'batch_size': 8,
    'in_channels': 32,
    'out_channels': 64,
    'height': 512,
    'width': 512,
    'kernel_size': (5, 9),
    'stride': 1,
    'padding': (2, 4),
    'dilation': (2, 3),
}


def create_inputs(dtype=jnp.float32):
    k_x, k_w = jax.random.split(jax.random.key(0))
    c = CONFIG
    x = jax.random.uniform(
        k_x, (c['batch_size'], c['in_channels'], c['height'], c['width']), dtype)
    kh, kw = c['kernel_size']
    w = jax.random.normal(
        k_w, (kh, kw, c['in_channels'], c['out_channels']), dtype) * 0.02
    return x, w


def workload(x, w):
    s = CONFIG['stride']
    ph, pw = CONFIG['padding']
    dh, dw = CONFIG['dilation']
    x_nhwc = jnp.transpose(x, (0, 2, 3, 1))
    y = lax.conv_general_dilated(
        x_nhwc, w,
        window_strides=(s, s),
        padding=((ph, ph), (pw, pw)),
        rhs_dilation=(dh, dw),
        dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
    )
    return jnp.transpose(y, (0, 3, 1, 2))
