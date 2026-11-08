"""Conv2d (square input, asymmetric 5x9 kernel, stride 1, no padding, no bias)."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': '62_conv_standard_2D__square_input__asymmetric_kernel',
    'batch_size': 8,
    'in_channels': 32,
    'out_channels': 64,
    'kernel_h': 5,
    'kernel_w': 9,
    'height': 512,
    'width': 512,
    'stride': 1,
    'padding': 0,
    'dilation': 1,
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    kx, kw = jax.random.split(key)
    x = jax.random.uniform(
        kx, (CONFIG['batch_size'], CONFIG['in_channels'], CONFIG['height'], CONFIG['width']), dtype)
    w = jax.random.normal(
        kw, (CONFIG['out_channels'], CONFIG['in_channels'], CONFIG['kernel_h'], CONFIG['kernel_w']), dtype) * 0.02
    return x, w


def workload(x, w):
    s = CONFIG['stride']
    p = CONFIG['padding']
    d = CONFIG['dilation']
    x_nhwc = jnp.transpose(x, (0, 2, 3, 1))
    w_hwio = jnp.transpose(w, (2, 3, 1, 0))
    y = lax.conv_general_dilated(
        x_nhwc, w_hwio,
        window_strides=(s, s),
        padding=((p, p), (p, p)),
        rhs_dilation=(d, d),
        dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
        precision=lax.Precision.HIGHEST,
    )
    return jnp.transpose(y, (0, 3, 1, 2))
