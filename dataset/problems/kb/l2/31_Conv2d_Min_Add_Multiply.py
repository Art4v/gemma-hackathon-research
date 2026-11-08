"""Conv2d followed by minimum with constant, bias addition, and scaling."""

import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': '31_Conv2d_Min_Add_Multiply',
    'batch_size': 128,
    'in_channels': 128,
    'out_channels': 128,
    'height': 128,
    'width': 128,
    'kernel_size': 3,
    'constant_value': 0.5,
    'scaling_factor': 2.0,
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
    conv_weight = jax.random.normal(k2, (c_out, c_in, k, k), dtype=dtype) * 0.02
    conv_bias = jax.random.normal(k3, (c_out,), dtype=dtype) * 0.02
    extra_bias = jax.random.normal(k4, (c_out, 1, 1), dtype=dtype) * 0.02
    return x, conv_weight, conv_bias, extra_bias


def workload(x, conv_weight, conv_bias, extra_bias):
    x_nhwc = jnp.transpose(x, (0, 2, 3, 1))
    w_hwio = jnp.transpose(conv_weight, (2, 3, 1, 0))
    out_nhwc = lax.conv_general_dilated(
        x_nhwc,
        w_hwio,
        window_strides=(1, 1),
        padding='VALID',
        dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
    )
    out_nhwc = out_nhwc + conv_bias.reshape(1, 1, 1, -1)
    out = jnp.transpose(out_nhwc, (0, 3, 1, 2))
    out = jnp.minimum(out, jnp.asarray(CONFIG['constant_value'], dtype=out.dtype))
    out = out + extra_bias
    out = out * CONFIG['scaling_factor']
    return out
