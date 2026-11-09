"""[bf16 variant] 3D convolution with asymmetric input and asymmetric kernel."""

import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': '66_conv_standard_3D__asymmetric_input__asymmetric_kernel',
    'batch_size': 16,
    'in_channels': 16,
    'out_channels': 64,
    'depth': 32,
    'height': 128,
    'width': 128,
    'kernel_size': (3, 5, 7),
    'stride': (1, 1, 1),
    'padding': (0, 0, 0),
    'dilation': (1, 1, 1),
    'groups': 1,
}


def create_inputs(dtype=jnp.bfloat16):
    k1, k2 = jax.random.split(jax.random.key(0), 2)
    b = CONFIG['batch_size']
    c_in = CONFIG['in_channels']
    c_out = CONFIG['out_channels']
    d = CONFIG['depth']
    h = CONFIG['height']
    w = CONFIG['width']
    kd, kh, kw = CONFIG['kernel_size']
    groups = CONFIG['groups']

    x = jax.random.uniform(k1, (b, c_in, d, h, w), dtype=dtype)
    weight = jax.random.normal(k2, (c_out, c_in // groups, kd, kh, kw), dtype=dtype) * 0.02
    return (x, weight)


def workload(x, weight):
    stride = CONFIG['stride']
    padding = tuple((p, p) for p in CONFIG['padding'])
    dilation = CONFIG['dilation']
    groups = CONFIG['groups']

    x_ndhwc = jnp.transpose(x, (0, 2, 3, 4, 1))
    w_dhwio = jnp.transpose(weight, (2, 3, 4, 1, 0))
    out_ndhwc = lax.conv_general_dilated(
        x_ndhwc,
        w_dhwio,
        window_strides=stride,
        padding=padding,
        rhs_dilation=dilation,
        dimension_numbers=('NDHWC', 'DHWIO', 'NDHWC'),
        feature_group_count=groups,
    )
    return jnp.transpose(out_ndhwc, (0, 4, 1, 2, 3))
