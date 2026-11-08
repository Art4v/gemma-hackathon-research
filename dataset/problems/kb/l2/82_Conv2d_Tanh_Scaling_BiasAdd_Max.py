"""2D convolution followed by tanh activation, scaling, bias addition, and 2D max pooling."""

import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': '82_Conv2d_Tanh_Scaling_BiasAdd_Max',
    'batch_size': 128,
    'in_channels': 8,
    'out_channels': 64,
    'height': 256,
    'width': 256,
    'kernel_size': 3,
    'scaling_factor': 2.0,
    'pool_kernel_size': 4,
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    k1, k2, k3, k4 = jax.random.split(key, 4)
    b = CONFIG['batch_size']
    c_in = CONFIG['in_channels']
    c_out = CONFIG['out_channels']
    h = CONFIG['height']
    w = CONFIG['width']
    k = CONFIG['kernel_size']

    x = jax.random.uniform(k1, (b, c_in, h, w), dtype=dtype)
    weight = jax.random.normal(k2, (c_out, c_in, k, k), dtype=dtype) * 0.02
    conv_bias = jax.random.normal(k3, (c_out,), dtype=dtype) * 0.02
    bias = jax.random.normal(k4, (c_out, 1, 1), dtype=dtype) * 0.02
    return x, weight, conv_bias, bias


def workload(x, weight, conv_bias, bias):
    p = CONFIG['pool_kernel_size']
    scale = CONFIG['scaling_factor']

    x_nhwc = jnp.transpose(x, (0, 2, 3, 1))
    w_hwio = jnp.transpose(weight, (2, 3, 1, 0))

    out = lax.conv_general_dilated(
        x_nhwc,
        w_hwio,
        window_strides=(1, 1),
        padding='VALID',
        dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
    )
    out = out + jnp.reshape(conv_bias, (1, 1, 1, -1))
    out = jnp.tanh(out)
    out = out * scale
    out = out + jnp.reshape(bias, (1, 1, 1, -1))

    out = lax.reduce_window(
        out,
        -jnp.inf,
        lax.max,
        window_dimensions=(1, p, p, 1),
        window_strides=(1, p, p, 1),
        padding='VALID',
    )
    return jnp.transpose(out, (0, 3, 1, 2))
