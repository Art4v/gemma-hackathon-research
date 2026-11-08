"""3D standard convolution with square input and asymmetric kernel."""

import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': '60_conv_standard_3D__square_input__asymmetric_kernel',
    'batch_size': 32,
    'in_channels': 16,
    'out_channels': 64,
    'width': 64,
    'height': 64,
    'depth': 64,
    'kernel_size': (3, 5, 7),
    'stride': (1, 1, 1),
    'padding': ((0, 0), (0, 0), (0, 0)),
    'dilation': (1, 1, 1),
    'groups': 1,
}


def create_inputs(dtype=jnp.float32):
    k1, k2 = jax.random.split(jax.random.key(0), 2)
    x_shape = (
        CONFIG['batch_size'],
        CONFIG['in_channels'],
        CONFIG['width'],
        CONFIG['height'],
        CONFIG['depth'],
    )
    w_shape = (
        CONFIG['out_channels'],
        CONFIG['in_channels'] // CONFIG['groups'],
        CONFIG['kernel_size'][0],
        CONFIG['kernel_size'][1],
        CONFIG['kernel_size'][2],
    )
    x = jax.random.uniform(k1, x_shape, dtype=dtype)
    weight = jax.random.normal(k2, w_shape, dtype=dtype) * 0.02
    return (x, weight)


def workload(x, weight):
    x_ndhwc = jnp.transpose(x, (0, 2, 3, 4, 1))
    w_dhwio = jnp.transpose(weight, (2, 3, 4, 1, 0))
    out_ndhwc = lax.conv_general_dilated(
        lhs=x_ndhwc,
        rhs=w_dhwio,
        window_strides=CONFIG['stride'],
        padding=CONFIG['padding'],
        rhs_dilation=CONFIG['dilation'],
        dimension_numbers=('NDHWC', 'DHWIO', 'NDHWC'),
        feature_group_count=CONFIG['groups'],
    )
    return jnp.transpose(out_ndhwc, (0, 4, 1, 2, 3))
