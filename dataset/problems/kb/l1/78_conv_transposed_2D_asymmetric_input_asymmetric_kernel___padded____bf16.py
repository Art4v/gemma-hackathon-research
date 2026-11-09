"""[bf16 variant] ConvTranspose2d (asymmetric input/kernel, padded, no bias)."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': '78_conv_transposed_2D_asymmetric_input_asymmetric_kernel___padded__',
    'batch_size': 8,
    'in_channels': 32,
    'out_channels': 32,
    'kernel_size': (3, 7),
    'height': 512,
    'width': 1024,
    'stride': (1, 1),
    'padding': (1, 3),
    'output_padding': (0, 0),
    'dilation': (1, 1),
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    kx, kw = jax.random.split(key)
    x = jax.random.uniform(
        kx, (CONFIG['batch_size'], CONFIG['in_channels'], CONFIG['height'], CONFIG['width']), dtype)
    kh, kwid = CONFIG['kernel_size']
    w = jax.random.normal(
        kw, (CONFIG['in_channels'], CONFIG['out_channels'], kh, kwid), dtype) * 0.02
    return x, w


def workload(x, w):
    # x: (N, C_in, H, W); w: (C_in, C_out, kH, kW) PyTorch ConvTranspose2d layout
    kh, kwid = w.shape[2], w.shape[3]
    sh, sw = CONFIG['stride']
    ph, pw = CONFIG['padding']
    oph, opw = CONFIG['output_padding']
    dh, dw = CONFIG['dilation']
    x_nhwc = jnp.transpose(x, (0, 2, 3, 1))
    w_hwio = jnp.transpose(jnp.flip(w, axis=(2, 3)), (2, 3, 0, 1))
    pad_h = dh * (kh - 1) - ph
    pad_w = dw * (kwid - 1) - pw
    y = lax.conv_general_dilated(
        x_nhwc, w_hwio,
        window_strides=(1, 1),
        padding=((pad_h, pad_h + oph), (pad_w, pad_w + opw)),
        lhs_dilation=(sh, sw),
        rhs_dilation=(dh, dw),
        dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
    )
    return jnp.transpose(y, (0, 3, 1, 2))
