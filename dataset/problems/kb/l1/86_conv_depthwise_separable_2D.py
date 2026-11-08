"""Depthwise conv2d (groups=C) followed by pointwise 1x1 conv2d, no bias."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': '86_conv_depthwise_separable_2D',
    'batch_size': 8,
    'in_channels': 128,
    'out_channels': 256,
    'kernel_size': 3,
    'height': 256,
    'width': 256,
    'stride': 1,
    'padding': 1,
    'dilation': 1,
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    kx, kd, kp = jax.random.split(key, 3)
    B = CONFIG['batch_size']
    C = CONFIG['in_channels']
    O = CONFIG['out_channels']
    K = CONFIG['kernel_size']
    H = CONFIG['height']
    W = CONFIG['width']
    x = jax.random.uniform(kx, (B, C, H, W), dtype=dtype)
    w_dw = jax.random.normal(kd, (K, K, 1, C), dtype) * 0.02
    w_pw = jax.random.normal(kp, (C, O), dtype) * 0.02
    return x, w_dw, w_pw


def workload(x, w_dw, w_pw):
    s = CONFIG['stride']
    p = CONFIG['padding']
    d = CONFIG['dilation']
    C = x.shape[1]
    xh = jnp.transpose(x, (0, 2, 3, 1))
    y = lax.conv_general_dilated(
        xh, w_dw,
        window_strides=(s, s),
        padding=((p, p), (p, p)),
        rhs_dilation=(d, d),
        dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
        feature_group_count=C,
        precision=lax.Precision.HIGHEST,
    )
    z = jnp.einsum('nhwc,co->nhwo', y, w_pw, precision=lax.Precision.HIGHEST)
    return jnp.transpose(z, (0, 3, 1, 2))
