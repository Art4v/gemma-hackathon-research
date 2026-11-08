"""2D convolution followed by HardSwish and ReLU activations."""

import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': '69_Conv2d_HardSwish_ReLU',
    'batch_size': 64,
    'in_channels': 32,
    'out_channels': 128,
    'height': 256,
    'width': 256,
    'kernel_size': 3,
    'stride': 1,
    'padding': 'VALID',
}

def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    k1, k2, k3 = jax.random.split(key, 3)
    x = jax.random.uniform(
        k1,
        (CONFIG['batch_size'], CONFIG['in_channels'], CONFIG['height'], CONFIG['width']),
        dtype=dtype,
    )
    weight = jax.random.normal(
        k2,
        (CONFIG['out_channels'], CONFIG['in_channels'], CONFIG['kernel_size'], CONFIG['kernel_size']),
        dtype=dtype,
    ) * 0.02
    bias = jax.random.normal(
        k3,
        (CONFIG['out_channels'],),
        dtype=dtype,
    ) * 0.02
    return (x, weight, bias)

def workload(x, weight, bias):
    x_nhwc = jnp.transpose(x, (0, 2, 3, 1))
    w_hwio = jnp.transpose(weight, (2, 3, 1, 0))
    out_nhwc = lax.conv_general_dilated(
        x_nhwc,
        w_hwio,
        window_strides=(CONFIG['stride'], CONFIG['stride']),
        padding=CONFIG['padding'],
        dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
    )
    out_nhwc = out_nhwc + bias
    out_nhwc = jax.nn.hard_swish(out_nhwc)
    out_nhwc = jax.nn.relu(out_nhwc)
    out_nchw = jnp.transpose(out_nhwc, (0, 3, 1, 2))
    return out_nchw
