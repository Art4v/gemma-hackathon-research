"""[bf16 variant] ConvTranspose2d (asymmetric input, asymmetric kernel, no bias)."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': '69_conv_transposed_2D__asymmetric_input__asymmetric_kernel',
    'batch_size': 64,
    'in_channels': 64,
    'out_channels': 128,
    'kernel_size': (3, 5),
    'height_in': 128,
    'width_in': 256,
    'stride': (1, 1),
    'padding': (0, 0),
    'output_padding': (0, 0),
    'dilation': (1, 1),
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    kx, kw = jax.random.split(key)
    x = jax.random.uniform(
        kx,
        (CONFIG['batch_size'], CONFIG['in_channels'], CONFIG['height_in'], CONFIG['width_in']),
        dtype,
    )
    kh, kwid = CONFIG['kernel_size']
    w = jax.random.normal(kw, (CONFIG['in_channels'], CONFIG['out_channels'], kh, kwid), dtype) * 0.02
    return x, w


def workload(x, w):
    # x: (N, Cin, H, W); w: PyTorch ConvTranspose layout (Cin, Cout, kH, kW)
    stride = CONFIG['stride']
    padding = CONFIG['padding']
    output_padding = CONFIG['output_padding']
    dilation = CONFIG['dilation']
    kH, kW = w.shape[2], w.shape[3]

    x_nhwc = jnp.transpose(x, (0, 2, 3, 1))
    # flip spatially, to HWIO (kH, kW, Cin, Cout)
    k = jnp.transpose(jnp.flip(w, axis=(2, 3)), (2, 3, 0, 1))

    pads = []
    for kk, s_p, s_op, s_d in zip((kH, kW), padding, output_padding, dilation):
        lo = s_d * (kk - 1) - s_p
        hi = s_d * (kk - 1) - s_p + s_op
        pads.append((lo, hi))

    y = lax.conv_general_dilated(
        x_nhwc,
        k,
        window_strides=(1, 1),
        padding=pads,
        lhs_dilation=stride,
        rhs_dilation=dilation,
        dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
    )
    return jnp.transpose(y, (0, 3, 1, 2))
