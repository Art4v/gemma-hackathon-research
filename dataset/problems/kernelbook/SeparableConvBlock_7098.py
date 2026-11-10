"""Depthwise separable convolution: depthwise Conv2d (groups=channels, no bias) -> 1x1 pointwise Conv2d -> bias."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'SeparableConvBlock_7098',
    'batch': 8,
    'in_channels': 128,
    'out_channels': 128,
    'height': 256,
    'width': 256,
    'kernel_size': 4,
    'stride': 1,
    'padding': 0,
    'dilation': 1,
    'weight_scale': 0.02,
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    k_x, k_depthwise, k_pointwise, k_bias = jax.random.split(key, 4)
    batch = CONFIG['batch']
    in_channels = CONFIG['in_channels']
    out_channels = CONFIG['out_channels']
    height = CONFIG['height']
    width = CONFIG['width']
    kernel_size = CONFIG['kernel_size']
    scale = CONFIG['weight_scale']
    x = jax.random.uniform(k_x, (batch, in_channels, height, width), dtype)
    depthwise_w = jax.random.normal(
        k_depthwise, (in_channels, 1, kernel_size, kernel_size), dtype) * scale
    pointwise_w = jax.random.normal(
        k_pointwise, (out_channels, in_channels, 1, 1), dtype) * scale
    pointwise_b = jax.random.normal(k_bias, (out_channels,), dtype) * scale
    return x, depthwise_w, pointwise_w, pointwise_b


def workload(x, depthwise_w, pointwise_w, pointwise_b):
    stride = (CONFIG['stride'], CONFIG['stride'])
    dilation = (CONFIG['dilation'], CONFIG['dilation'])
    pad = CONFIG['padding']
    padding = [(pad, pad), (pad, pad)]
    dims = ('NHWC', 'OIHW', 'NHWC')
    y = jnp.transpose(x, (0, 2, 3, 1))
    y = lax.conv_general_dilated(
        y,
        depthwise_w,
        window_strides=stride,
        padding=padding,
        rhs_dilation=dilation,
        dimension_numbers=dims,
        feature_group_count=x.shape[1],
    )
    y = lax.conv_general_dilated(
        y,
        pointwise_w,
        window_strides=(1, 1),
        padding='VALID',
        dimension_numbers=dims,
    )
    y = y + pointwise_b
    return jnp.transpose(y, (0, 3, 1, 2))
