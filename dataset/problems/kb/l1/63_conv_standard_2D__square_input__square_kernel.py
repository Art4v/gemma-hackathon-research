"""Standard 2D convolution with square input and square kernel."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': '63_conv_standard_2D__square_input__square_kernel',
    'batch_size': 8,
    'in_channels': 128,
    'out_channels': 128,
    'kernel_size': 3,
    'height': 256,
    'width': 256,
    'stride': 1,
    'padding': 0,
    'dilation': 1,
    'groups': 1,
}

def create_inputs(dtype=jnp.float32):
    k1, k2 = jax.random.split(jax.random.key(0), 2)
    x = jax.random.uniform(
        k1,
        (CONFIG['batch_size'], CONFIG['in_channels'], CONFIG['height'], CONFIG['width']),
        dtype=dtype,
    )
    weight = jax.random.normal(
        k2,
        (
            CONFIG['out_channels'],
            CONFIG['in_channels'] // CONFIG['groups'],
            CONFIG['kernel_size'],
            CONFIG['kernel_size'],
        ),
        dtype=dtype,
    ) * 0.02
    return (x, weight)

def workload(x, weight):
    stride = CONFIG['stride']
    padding = CONFIG['padding']
    dilation = CONFIG['dilation']
    groups = CONFIG['groups']

    x_nhwc = jnp.transpose(x, (0, 2, 3, 1))
    w_hwio = jnp.transpose(weight, (2, 3, 1, 0))

    out_nhwc = lax.conv_general_dilated(
        lhs=x_nhwc,
        rhs=w_hwio,
        window_strides=(stride, stride),
        padding=((padding, padding), (padding, padding)),
        rhs_dilation=(dilation, dilation),
        dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
        feature_group_count=groups,
    )
    return jnp.transpose(out_nhwc, (0, 3, 1, 2))
