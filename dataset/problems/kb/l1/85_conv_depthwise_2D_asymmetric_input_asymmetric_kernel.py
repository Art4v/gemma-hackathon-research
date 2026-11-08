"""Depthwise 2D convolution with asymmetric input and asymmetric kernel."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': '85_conv_depthwise_2D_asymmetric_input_asymmetric_kernel',
    'batch_size': 32,
    'in_channels': 128,
    'out_channels': 128,
    'height': 128,
    'width': 256,
    'kernel_size_h': 3,
    'kernel_size_w': 7,
    'stride_h': 1,
    'stride_w': 1,
    'padding_h': 0,
    'padding_w': 0,
    'dilation_h': 1,
    'dilation_w': 1,
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
        (CONFIG['in_channels'], 1, CONFIG['kernel_size_h'], CONFIG['kernel_size_w']),
        dtype=dtype,
    ) * 0.02
    return (x, weight)


def workload(x, weight):
    x_nhwc = jnp.transpose(x, (0, 2, 3, 1))
    out_nhwc = lax.conv_general_dilated(
        lhs=x_nhwc,
        rhs=weight,
        window_strides=(CONFIG['stride_h'], CONFIG['stride_w']),
        padding=(
            (CONFIG['padding_h'], CONFIG['padding_h']),
            (CONFIG['padding_w'], CONFIG['padding_w']),
        ),
        rhs_dilation=(CONFIG['dilation_h'], CONFIG['dilation_w']),
        dimension_numbers=('NHWC', 'OIHW', 'NHWC'),
        feature_group_count=x.shape[1],
    )
    return jnp.transpose(out_nhwc, (0, 3, 1, 2))
