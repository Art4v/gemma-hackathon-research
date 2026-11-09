"""[bf16 variant] Transposed 2D convolution with asymmetric input and asymmetric kernel."""

import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': '69_conv_transposed_2D__asymmetric_input__asymmetric_kernel',
    'batch_size': 64,
    'in_channels': 64,
    'out_channels': 128,
    'height_in': 128,
    'width_in': 256,
    'kernel_size': (3, 5),
    'stride': (1, 1),
    'padding': (0, 0),
    'output_padding': (0, 0),
    'dilation': (1, 1),
    'groups': 1,
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    k1, k2 = jax.random.split(key, 2)
    B = CONFIG['batch_size']
    C_in = CONFIG['in_channels']
    C_out = CONFIG['out_channels']
    H = CONFIG['height_in']
    W = CONFIG['width_in']
    kH, kW = CONFIG['kernel_size']
    G = CONFIG['groups']

    x = jax.random.uniform(k1, (B, C_in, H, W), dtype=dtype)
    weight = jax.random.normal(k2, (C_in, C_out // G, kH, kW), dtype=dtype) * 0.02
    return (x, weight)


def workload(x, weight):
    stride_h, stride_w = CONFIG['stride']
    pad_h, pad_w = CONFIG['padding']
    out_pad_h, out_pad_w = CONFIG['output_padding']
    dil_h, dil_w = CONFIG['dilation']
    groups = CONFIG['groups']

    kH, kW = weight.shape[2], weight.shape[3]
    C_in = weight.shape[0]
    C_out_per_g = weight.shape[1]

    x_nhwc = jnp.transpose(x, (0, 2, 3, 1))

    w_flipped = jnp.flip(weight, axis=(2, 3))
    w_grouped = jnp.reshape(w_flipped, (groups, C_in // groups, C_out_per_g, kH, kW))
    w_hwio = jnp.transpose(w_grouped, (3, 4, 1, 0, 2))
    w_hwio = jnp.reshape(w_hwio, (kH, kW, C_in // groups, groups * C_out_per_g))

    eff_kH = (kH - 1) * dil_h + 1
    eff_kW = (kW - 1) * dil_w + 1
    pad_top = eff_kH - 1 - pad_h
    pad_bottom = eff_kH - 1 - pad_h + out_pad_h
    pad_left = eff_kW - 1 - pad_w
    pad_right = eff_kW - 1 - pad_w + out_pad_w

    out_nhwc = lax.conv_general_dilated(
        lhs=x_nhwc,
        rhs=w_hwio,
        window_strides=(1, 1),
        padding=((pad_top, pad_bottom), (pad_left, pad_right)),
        lhs_dilation=(stride_h, stride_w),
        rhs_dilation=(dil_h, dil_w),
        dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
        feature_group_count=groups,
    )
    return jnp.transpose(out_nhwc, (0, 3, 1, 2))
