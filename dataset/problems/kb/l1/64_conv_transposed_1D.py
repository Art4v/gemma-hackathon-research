"""1D transposed convolution (ConvTranspose1d)."""

import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': '64_conv_transposed_1D',
    'batch_size': 16,
    'in_channels': 128,
    'out_channels': 128,
    'kernel_size': 3,
    'length': 65536,
    'stride': 1,
    'padding': 0,
    'output_padding': 0,
    'groups': 1,
}


def create_inputs(dtype=jnp.float32):
    k1, k2 = jax.random.split(jax.random.key(0), 2)
    B = CONFIG['batch_size']
    C_in = CONFIG['in_channels']
    C_out = CONFIG['out_channels']
    K = CONFIG['kernel_size']
    L = CONFIG['length']
    G = CONFIG['groups']

    x = jax.random.uniform(k1, (B, C_in, L), dtype=dtype)
    weight = jax.random.normal(k2, (C_in, C_out // G, K), dtype=dtype) * 0.02
    return (x, weight)


def workload(x, weight):
    stride = CONFIG['stride']
    padding = CONFIG['padding']
    output_padding = CONFIG['output_padding']
    groups = CONFIG['groups']
    kernel_size = CONFIG['kernel_size']
    in_channels = x.shape[1]
    out_channels = weight.shape[1] * groups

    x_nhc = jnp.transpose(x, (0, 2, 1))

    w = jnp.reshape(weight, (groups, in_channels // groups, out_channels // groups, kernel_size))
    w = jnp.transpose(w, (3, 1, 0, 2))
    w = jnp.reshape(w, (kernel_size, in_channels // groups, out_channels))
    w = jnp.flip(w, axis=0)

    pad_before = kernel_size - 1 - padding
    pad_after = kernel_size - 1 - padding + output_padding

    out_nhc = lax.conv_general_dilated(
        lhs=x_nhc,
        rhs=w,
        window_strides=(1,),
        padding=[(pad_before, pad_after)],
        lhs_dilation=(stride,),
        dimension_numbers=('NHC', 'HIO', 'NHC'),
        feature_group_count=groups,
    )
    return jnp.transpose(out_nhc, (0, 2, 1))
