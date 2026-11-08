"""ConvTranspose2d (square input, asymmetric 3x7 kernel, stride 1, no bias)."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': '65_conv_transposed_2D__square_input__asymmetric_kernel',
    'batch': 8,
    'in_channels': 64,
    'out_channels': 64,
    'height': 512,
    'width': 512,
    'kernel_h': 3,
    'kernel_w': 7,
    'stride': 1,
    'padding': 0,
    'output_padding': 0,
    'dilation': 1,
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    kx, kw = jax.random.split(key)
    c = CONFIG
    x = jax.random.uniform(kx, (c['batch'], c['in_channels'], c['height'], c['width']), dtype)
    # PyTorch ConvTranspose2d weight layout: (in_channels, out_channels, kH, kW)
    w = jax.random.normal(kw, (c['in_channels'], c['out_channels'], c['kernel_h'], c['kernel_w']), dtype) * 0.02
    return x, w


def workload(x, w):
    s = CONFIG['stride']
    p = CONFIG['padding']
    op = CONFIG['output_padding']
    d = CONFIG['dilation']
    kh, kw = w.shape[2], w.shape[3]
    # (Cin, Cout, kH, kW) -> flipped HWIO (kH, kW, Cin, Cout)
    k = jnp.transpose(jnp.flip(w, axis=(2, 3)), (2, 3, 0, 1))
    x_nhwc = jnp.transpose(x, (0, 2, 3, 1))
    pad_h = (d * (kh - 1) - p, d * (kh - 1) - p + op)
    pad_w = (d * (kw - 1) - p, d * (kw - 1) - p + op)
    y = lax.conv_general_dilated(
        x_nhwc, k,
        window_strides=(1, 1),
        padding=(pad_h, pad_w),
        lhs_dilation=(s, s),
        rhs_dilation=(d, d),
        dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
    )
    return jnp.transpose(y, (0, 3, 1, 2))
