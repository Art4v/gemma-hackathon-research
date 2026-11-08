"""Conv3d (standard, square input, asymmetric kernel, no bias)."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': '60_conv_standard_3D__square_input__asymmetric_kernel',
    'batch_size': 8,
    'in_channels': 16,
    'out_channels': 64,
    'kernel_size': (3, 5, 7),
    'width': 48,
    'height': 48,
    'depth': 48,
    'stride': 1,
    'padding': 0,
    'dilation': 1,
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    kx, kw = jax.random.split(key)
    x = jax.random.uniform(
        kx,
        (CONFIG['batch_size'], CONFIG['in_channels'],
         CONFIG['width'], CONFIG['height'], CONFIG['depth']),
        dtype=dtype)
    k0, k1, k2 = CONFIG['kernel_size']
    w = jax.random.normal(
        kw, (k0, k1, k2, CONFIG['in_channels'], CONFIG['out_channels']), dtype) * 0.02
    return x, w


def workload(x, w):
    s = CONFIG['stride']
    p = CONFIG['padding']
    d = CONFIG['dilation']
    x_nl = jnp.transpose(x, (0, 2, 3, 4, 1))  # NCDHW -> NDHWC
    y = lax.conv_general_dilated(
        x_nl, w,
        window_strides=(s, s, s),
        padding=[(p, p), (p, p), (p, p)],
        rhs_dilation=(d, d, d),
        dimension_numbers=('NDHWC', 'DHWIO', 'NDHWC'),
    )
    return jnp.transpose(y, (0, 4, 1, 2, 3))  # NDHWC -> NCDHW
