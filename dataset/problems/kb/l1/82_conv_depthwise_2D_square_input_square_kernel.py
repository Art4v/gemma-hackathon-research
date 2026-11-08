"""Depthwise 2D convolution with square input and square kernel."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': '82_conv_depthwise_2D_square_input_square_kernel',
    'batch_size': 16,
    'in_channels': 128,
    'height': 512,
    'width': 512,
    'kernel_size': 3,
    'stride': 1,
    'padding': 0,
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
        (CONFIG['in_channels'], 1, CONFIG['kernel_size'], CONFIG['kernel_size']),
        dtype=dtype,
    ) * 0.02
    return (x, weight)

def workload(x, weight):
    stride = CONFIG['stride']
    padding = CONFIG['padding']
    in_channels = x.shape[1]
    x_nhwc = jnp.transpose(x, (0, 2, 3, 1))
    w_hwio = jnp.transpose(weight, (2, 3, 1, 0))
    out_nhwc = lax.conv_general_dilated(
        x_nhwc,
        w_hwio,
        window_strides=(stride, stride),
        padding=((padding, padding), (padding, padding)),
        dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
        feature_group_count=in_channels,
    )
    return jnp.transpose(out_nhwc, (0, 3, 1, 2))
