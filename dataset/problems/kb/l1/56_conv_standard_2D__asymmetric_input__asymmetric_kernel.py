"""Standard 2D convolution (asymmetric input, asymmetric 5x7 kernel, no bias)."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': '56_conv_standard_2D__asymmetric_input__asymmetric_kernel',
    'batch_size': 8,
    'in_channels': 64,
    'out_channels': 128,
    'kernel_h': 5,
    'kernel_w': 7,
    'height': 512,
    'width': 256,
    'stride': (1, 1),
    'padding': (0, 0),
    'dilation': (1, 1),
    'groups': 1,
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    kx, kw = jax.random.split(key)
    x = jax.random.uniform(
        kx,
        (CONFIG['batch_size'], CONFIG['in_channels'], CONFIG['height'], CONFIG['width']),
        dtype=dtype,
    )
    w = jax.random.normal(
        kw,
        (CONFIG['kernel_h'], CONFIG['kernel_w'],
         CONFIG['in_channels'] // CONFIG['groups'], CONFIG['out_channels']),
        dtype=dtype,
    ) * 0.02
    return x, w


def workload(x, w):
    ph, pw = CONFIG['padding']
    x_nhwc = jnp.transpose(x, (0, 2, 3, 1))
    y = lax.conv_general_dilated(
        x_nhwc,
        w,
        window_strides=CONFIG['stride'],
        padding=((ph, ph), (pw, pw)),
        rhs_dilation=CONFIG['dilation'],
        dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
        feature_group_count=CONFIG['groups'],
    )
    return jnp.transpose(y, (0, 3, 1, 2))
