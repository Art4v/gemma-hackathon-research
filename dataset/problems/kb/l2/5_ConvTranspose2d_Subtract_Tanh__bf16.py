"""[bf16 variant] ConvTranspose2d (stride 2, pad 1, output_pad 1) -> subtract per-channel bias -> tanh."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': '5_ConvTranspose2d_Subtract_Tanh',
    'batch_size': 32,
    'in_channels': 128,
    'out_channels': 128,
    'height': 128,
    'width': 128,
    'kernel_size': 4,
    'stride': 2,
    'padding': 1,
    'output_padding': 1,
}


def create_inputs(dtype=jnp.bfloat16):
    c = CONFIG
    key = jax.random.key(0)
    kx, kw, kb, kbias = jax.random.split(key, 4)
    x = jax.random.uniform(kx, (c['batch_size'], c['in_channels'], c['height'], c['width']), dtype)
    w = jax.random.normal(kw, (c['in_channels'], c['out_channels'], c['kernel_size'], c['kernel_size']), dtype) * 0.02
    b_conv = jax.random.normal(kb, (c['out_channels'],), dtype) * 0.02
    bias = jax.random.normal(kbias, (c['out_channels'], 1, 1), dtype)
    return x, w, b_conv, bias


def workload(x, w, b_conv, bias):
    s = CONFIG['stride']
    p = CONFIG['padding']
    op = CONFIG['output_padding']
    kh, kw = w.shape[2], w.shape[3]
    x_nhwc = jnp.transpose(x, (0, 2, 3, 1))
    # PyTorch ConvTranspose weight (I, O, kH, kW) -> flipped HWIO kernel
    k_hwio = jnp.transpose(w[:, :, ::-1, ::-1], (2, 3, 0, 1))
    pad = ((kh - 1 - p, kh - 1 - p + op), (kw - 1 - p, kw - 1 - p + op))
    y = lax.conv_general_dilated(
        x_nhwc, k_hwio,
        window_strides=(1, 1),
        padding=pad,
        lhs_dilation=(s, s),
        rhs_dilation=(1, 1),
        dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
    )
    y = y + b_conv
    y = jnp.transpose(y, (0, 3, 1, 2))
    y = y - bias
    return jnp.tanh(y)
