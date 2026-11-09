"""[bf16 variant] Transposed 3D convolution with asymmetric input and square kernel."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': '70_conv_transposed_3D__asymmetric_input__square_kernel',
    'batch_size': 8,
    'in_channels': 128,
    'out_channels': 64,
    'kernel_size': 3,
    'depth': 32,
    'height': 64,
    'width': 64,
    'stride': 1,
    'padding': 0,
    'output_padding': 0,
    'dilation': 1,
    'groups': 1,
}


def create_inputs(dtype=jnp.bfloat16):
    k1, k2 = jax.random.split(jax.random.key(0), 2)
    B = CONFIG['batch_size']
    C_in = CONFIG['in_channels']
    C_out = CONFIG['out_channels']
    K = CONFIG['kernel_size']
    D = CONFIG['depth']
    H = CONFIG['height']
    W = CONFIG['width']
    G = CONFIG['groups']

    x = jax.random.uniform(k1, (B, C_in, D, H, W), dtype=dtype)
    weight = jax.random.normal(k2, (C_in, C_out // G, K, K, K), dtype=dtype) * 0.02
    return (x, weight)


def workload(x, weight):
    stride = CONFIG['stride']
    padding = CONFIG['padding']
    output_padding = CONFIG['output_padding']
    dilation = CONFIG['dilation']
    groups = CONFIG['groups']

    s = (stride, stride, stride) if isinstance(stride, int) else stride
    p = (padding, padding, padding) if isinstance(padding, int) else padding
    op = (output_padding, output_padding, output_padding) if isinstance(output_padding, int) else output_padding
    d = (dilation, dilation, dilation) if isinstance(dilation, int) else dilation

    in_ch = weight.shape[0]
    out_ch_per_g = weight.shape[1]
    kD, kH, kW = weight.shape[2], weight.shape[3], weight.shape[4]
    k_shape = (kD, kH, kW)

    x_ndhwc = jnp.transpose(x, (0, 2, 3, 4, 1))

    w = jnp.flip(weight, axis=(2, 3, 4))
    if groups > 1:
        w = jnp.reshape(w, (groups, in_ch // groups, out_ch_per_g, kD, kH, kW))
        w = jnp.transpose(w, (0, 2, 1, 3, 4, 5))
        w = jnp.reshape(w, (groups * out_ch_per_g, in_ch // groups, kD, kH, kW))
        rhs_spec = 'OIDHW'
    else:
        rhs_spec = 'IODHW'

    jax_padding = []
    for i in range(3):
        k_eff = (k_shape[i] - 1) * d[i] + 1
        pad_before = k_eff - 1 - p[i]
        pad_after = k_eff - 1 - p[i] + op[i]
        jax_padding.append((pad_before, pad_after))

    out_ndhwc = lax.conv_general_dilated(
        lhs=x_ndhwc,
        rhs=w,
        window_strides=(1, 1, 1),
        padding=jax_padding,
        lhs_dilation=s,
        rhs_dilation=d,
        dimension_numbers=('NDHWC', rhs_spec, 'NDHWC'),
        feature_group_count=groups,
    )

    return jnp.transpose(out_ndhwc, (0, 4, 1, 2, 3))
