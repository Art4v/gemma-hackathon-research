"""3D convolution (asymmetric kernel 3x5x7, no bias, stride 1, no padding)."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': '66_conv_standard_3D__asymmetric_input__asymmetric_kernel',
    'batch_size': 16,
    'in_channels': 3,
    'out_channels': 64,
    'kernel_size': (3, 5, 7),
    'depth': 16,
    'height': 128,
    'width': 128,
    'stride': (1, 1, 1),
    'padding': (0, 0, 0),
    'dilation': (1, 1, 1),
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    kx, kw = jax.random.split(key)
    x = jax.random.uniform(
        kx,
        (CONFIG['batch_size'], CONFIG['in_channels'], CONFIG['depth'],
         CONFIG['height'], CONFIG['width']),
        dtype=dtype)
    kd, kh, kwid = CONFIG['kernel_size']
    w = jax.random.normal(
        kw, (kd, kh, kwid, CONFIG['in_channels'], CONFIG['out_channels']),
        dtype) * 0.02
    return x, w


def workload(x, w):
    # x: (N, C, D, H, W); w: (kd, kh, kw, Cin, Cout)
    x_ndhwc = jnp.transpose(x, (0, 2, 3, 4, 1))
    pad = [(p, p) for p in CONFIG['padding']]
    y = lax.conv_general_dilated(
        x_ndhwc, w,
        window_strides=CONFIG['stride'],
        padding=pad,
        rhs_dilation=CONFIG['dilation'],
        dimension_numbers=('NDHWC', 'DHWIO', 'NDHWC'),
        precision=lax.Precision.HIGHEST,
    )
    return jnp.transpose(y, (0, 4, 1, 2, 3))
