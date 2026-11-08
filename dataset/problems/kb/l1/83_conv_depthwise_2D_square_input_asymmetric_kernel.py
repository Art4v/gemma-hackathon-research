"""Depthwise Conv2d with asymmetric (K x 1) kernel, groups=in_channels, no bias."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': '83_conv_depthwise_2D_square_input_asymmetric_kernel',
    'batch_size': 64,
    'in_channels': 8,
    'kernel_size': 3,
    'height': 512,
    'width': 512,
    'stride': 1,
    'padding': 0,
    'dilation': 1,
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    kx, kw = jax.random.split(key)
    B, C = CONFIG['batch_size'], CONFIG['in_channels']
    H, W, K = CONFIG['height'], CONFIG['width'], CONFIG['kernel_size']
    x = jax.random.uniform(kx, (B, C, H, W), dtype=dtype)
    w = jax.random.normal(kw, (C, 1, K, 1), dtype=dtype) * 0.02
    return x, w


def workload(x, w):
    C = x.shape[1]
    s, p, d = CONFIG['stride'], CONFIG['padding'], CONFIG['dilation']
    x_nhwc = jnp.transpose(x, (0, 2, 3, 1))
    w_hwio = jnp.transpose(w, (2, 3, 1, 0))  # (K, 1, 1, C)
    y = lax.conv_general_dilated(
        x_nhwc, w_hwio,
        window_strides=(s, s),
        padding=((p, p), (p, p)),
        rhs_dilation=(d, d),
        dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
        feature_group_count=C,
    )
    return jnp.transpose(y, (0, 3, 1, 2))
