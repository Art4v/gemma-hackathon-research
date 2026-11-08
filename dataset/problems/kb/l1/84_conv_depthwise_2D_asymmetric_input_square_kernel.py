"""Depthwise 2D convolution (groups=in_channels, square kernel, no bias) on an asymmetric NCHW input."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': '84_conv_depthwise_2D_asymmetric_input_square_kernel',
    'batch_size': 8,
    'in_channels': 128,
    'out_channels': 128,
    'kernel_size': 3,
    'height_in': 256,
    'width_in': 512,
    'stride': 1,
    'padding': 0,
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    kx, kw = jax.random.split(key)
    x = jax.random.uniform(
        kx,
        (CONFIG['batch_size'], CONFIG['in_channels'], CONFIG['height_in'], CONFIG['width_in']),
        dtype,
    )
    k = CONFIG['kernel_size']
    w = jax.random.normal(kw, (k, k, 1, CONFIG['out_channels']), dtype) * 0.02
    return (x, w)


def workload(x, w):
    s = CONFIG['stride']
    p = CONFIG['padding']
    groups = x.shape[1]
    x_nhwc = jnp.transpose(x, (0, 2, 3, 1))
    y = lax.conv_general_dilated(
        x_nhwc,
        w,
        window_strides=(s, s),
        padding=[(p, p), (p, p)],
        dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
        feature_group_count=groups,
    )
    return jnp.transpose(y, (0, 3, 1, 2))
