"""Depthwise 2D convolution (asymmetric 3x7 kernel, groups=in_channels, no bias)."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': '85_conv_depthwise_2D_asymmetric_input_asymmetric_kernel',
    'batch_size': 32,
    'in_channels': 128,
    'height': 128,
    'width': 256,
    'kernel_size_h': 3,
    'kernel_size_w': 7,
    'stride_h': 1,
    'stride_w': 1,
    'padding_h': 0,
    'padding_w': 0,
    'dilation_h': 1,
    'dilation_w': 1,
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    kx, kw = jax.random.split(key)
    c = CONFIG
    x = jax.random.uniform(kx, (c['batch_size'], c['in_channels'], c['height'], c['width']), dtype)
    w = jax.random.normal(kw, (c['kernel_size_h'], c['kernel_size_w'], 1, c['in_channels']), dtype) * 0.02
    return x, w


def workload(x, w):
    c = CONFIG
    channels = x.shape[1]
    x_nhwc = jnp.transpose(x, (0, 2, 3, 1))
    y = lax.conv_general_dilated(
        x_nhwc,
        w,
        window_strides=(c['stride_h'], c['stride_w']),
        padding=((c['padding_h'], c['padding_h']), (c['padding_w'], c['padding_w'])),
        rhs_dilation=(c['dilation_h'], c['dilation_w']),
        dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
        feature_group_count=channels,
    )
    return jnp.transpose(y, (0, 3, 1, 2))
