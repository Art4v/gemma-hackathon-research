"""Standard 1D convolution (Conv1d)."""

import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': '67_conv_standard_1D',
    'batch_size': 32,
    'in_channels': 128,
    'out_channels': 128,
    'kernel_size': 3,
    'length': 131072,
    'stride': 1,
    'padding': 0,
    'dilation': 1,
    'groups': 1,
}


def create_inputs(dtype=jnp.float32):
    k1, k2 = jax.random.split(jax.random.key(0), 2)
    x = jax.random.uniform(
        k1,
        (CONFIG['batch_size'], CONFIG['in_channels'], CONFIG['length']),
        dtype=dtype,
    )
    weight = (
        jax.random.normal(
            k2,
            (
                CONFIG['out_channels'],
                CONFIG['in_channels'] // CONFIG['groups'],
                CONFIG['kernel_size'],
            ),
            dtype=dtype,
        )
        * 0.02
    )
    return (x, weight)


def workload(x, weight):
    x_nlc = jnp.transpose(x, (0, 2, 1))
    w_kio = jnp.transpose(weight, (2, 1, 0))
    out_nlc = lax.conv_general_dilated(
        lhs=x_nlc,
        rhs=w_kio,
        window_strides=(CONFIG['stride'],),
        padding=((CONFIG['padding'], CONFIG['padding']),),
        lhs_dilation=(1,),
        rhs_dilation=(CONFIG['dilation'],),
        dimension_numbers=('NWC', 'WIO', 'NWC'),
        feature_group_count=CONFIG['groups'],
    )
    return jnp.transpose(out_nlc, (0, 2, 1))
