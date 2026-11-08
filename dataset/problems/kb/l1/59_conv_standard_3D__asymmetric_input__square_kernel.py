"""Standard 3D convolution with asymmetric input and (k, k, 1) kernel."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': '59_conv_standard_3D__asymmetric_input__square_kernel',
    'batch_size': 16,
    'in_channels': 8,
    'out_channels': 128,
    'kernel_size': 3,
    'height': 256,
    'width': 256,
    'depth': 16,
    'stride': 1,
    'padding': 0,
    'dilation': 1,
    'groups': 1,
}


def create_inputs(dtype=jnp.float32):
    k1, k2 = jax.random.split(jax.random.key(0), 2)
    x = jax.random.uniform(
        k1,
        (
            CONFIG['batch_size'],
            CONFIG['in_channels'],
            CONFIG['height'],
            CONFIG['width'],
            CONFIG['depth'],
        ),
        dtype=dtype,
    )
    weight = (
        jax.random.normal(
            k2,
            (
                CONFIG['out_channels'],
                CONFIG['in_channels'] // CONFIG['groups'],
                CONFIG['kernel_size'],
                CONFIG['kernel_size'],
                1,
            ),
            dtype=dtype,
        )
        * 0.02
    )
    return (x, weight)


def workload(x, weight):
    stride = CONFIG['stride']
    padding = CONFIG['padding']
    dilation = CONFIG['dilation']
    groups = CONFIG['groups']

    x_ndhwc = jnp.transpose(x, (0, 2, 3, 4, 1))
    w_dhwio = jnp.transpose(weight, (2, 3, 4, 1, 0))

    out_ndhwc = lax.conv_general_dilated(
        lhs=x_ndhwc,
        rhs=w_dhwio,
        window_strides=(stride, stride, stride),
        padding=((padding, padding), (padding, padding), (padding, padding)),
        rhs_dilation=(dilation, dilation, dilation),
        dimension_numbers=('NDHWC', 'DHWIO', 'NDHWC'),
        feature_group_count=groups,
    )
    return jnp.transpose(out_ndhwc, (0, 4, 1, 2, 3))
