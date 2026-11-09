"""[bf16 variant] ComplexConv2d: two Conv2d(3x3, bias) combined as complex multiplication (real, imag)."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'ComplexConv2d_12323',
    'batch': 8,
    'in_channels': 128,
    'out_channels': 128,
    'height': 128,
    'width': 128,
    'kernel_size': 3,
    'stride': 1,
    'padding': 0,
    'dilation': 1,
}


def create_inputs(dtype=jnp.bfloat16):
    c = CONFIG
    keys = jax.random.split(jax.random.key(0), 6)
    x_shape = (c['batch'], c['in_channels'], c['height'], c['width'])
    w_shape = (c['kernel_size'], c['kernel_size'], c['in_channels'], c['out_channels'])
    input_r = jax.random.uniform(keys[0], x_shape, dtype)
    input_i = jax.random.uniform(keys[1], x_shape, dtype)
    w_r = jax.random.normal(keys[2], w_shape, dtype) * 0.02
    b_r = jax.random.normal(keys[3], (c['out_channels'],), dtype) * 0.02
    w_i = jax.random.normal(keys[4], w_shape, dtype) * 0.02
    b_i = jax.random.normal(keys[5], (c['out_channels'],), dtype) * 0.02
    return input_r, input_i, w_r, b_r, w_i, b_i


def _conv(x_nhwc, w, b):
    s, p, d = CONFIG['stride'], CONFIG['padding'], CONFIG['dilation']
    y = lax.conv_general_dilated(
        x_nhwc, w,
        window_strides=(s, s),
        padding=((p, p), (p, p)),
        rhs_dilation=(d, d),
        dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
    )
    return y + b


def workload(input_r, input_i, w_r, b_r, w_i, b_i):
    xr = jnp.transpose(input_r, (0, 2, 3, 1))
    xi = jnp.transpose(input_i, (0, 2, 3, 1))
    real = _conv(xr, w_r, b_r) - _conv(xi, w_i, b_i)
    imag = _conv(xi, w_r, b_r) + _conv(xr, w_i, b_i)
    return jnp.transpose(real, (0, 3, 1, 2)), jnp.transpose(imag, (0, 3, 1, 2))
