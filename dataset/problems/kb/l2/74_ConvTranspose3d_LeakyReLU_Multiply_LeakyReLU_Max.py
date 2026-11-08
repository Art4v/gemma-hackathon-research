"""3D transposed convolution followed by LeakyReLU, channel-wise multiplication, LeakyReLU, and 3D max pooling."""

import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': '74_ConvTranspose3d_LeakyReLU_Multiply_LeakyReLU_Max',
    'batch_size': 16,
    'in_channels': 64,
    'out_channels': 128,
    'depth': 16,
    'height': 64,
    'width': 64,
    'kernel_size': 3,
    'stride': 2,
    'padding': 1,
    'output_padding': 1,
    'negative_slope': 0.2,
    'max_pool_kernel_size': 2,
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    k_x, k_w, k_b, k_m = jax.random.split(key, 4)
    b = CONFIG['batch_size']
    c_in = CONFIG['in_channels']
    c_out = CONFIG['out_channels']
    d = CONFIG['depth']
    h = CONFIG['height']
    w = CONFIG['width']
    k = CONFIG['kernel_size']

    x = jax.random.uniform(k_x, (b, c_in, d, h, w), dtype=dtype)
    weight = jax.random.normal(k_w, (c_in, c_out, k, k, k), dtype=dtype) * 0.02
    bias = jax.random.normal(k_b, (c_out,), dtype=dtype) * 0.02
    multiplier = jax.random.normal(k_m, (c_out, 1, 1, 1), dtype=dtype) * 0.02
    return x, weight, bias, multiplier


def workload(x, weight, bias, multiplier):
    stride = CONFIG['stride']
    padding = CONFIG['padding']
    output_padding = CONFIG['output_padding']
    negative_slope = CONFIG['negative_slope']
    mp_k = CONFIG['max_pool_kernel_size']
    k = weight.shape[2]

    x_ndhwc = jnp.transpose(x, (0, 2, 3, 4, 1))
    w_dhwio = jnp.transpose(weight, (2, 3, 4, 0, 1))
    w_flipped = jnp.flip(w_dhwio, axis=(0, 1, 2))

    pad_before = k - 1 - padding
    pad_after = k - 1 - padding + output_padding
    pads = [(pad_before, pad_after)] * 3

    dn = lax.conv_dimension_numbers(
        x_ndhwc.shape,
        w_flipped.shape,
        ('NDHWC', 'DHWIO', 'NDHWC'),
    )
    out = lax.conv_general_dilated(
        lhs=x_ndhwc,
        rhs=w_flipped,
        window_strides=(1, 1, 1),
        padding=pads,
        lhs_dilation=(stride, stride, stride),
        dimension_numbers=dn,
    )
    out = out + bias.reshape((1, 1, 1, 1, -1))
    out = jnp.where(out >= 0, out, out * negative_slope)

    mult_ndhwc = jnp.transpose(multiplier, (1, 2, 3, 0)).reshape((1, 1, 1, 1, -1))
    out = out * mult_ndhwc
    out = jnp.where(out >= 0, out, out * negative_slope)

    out = lax.reduce_window(
        out,
        -jnp.inf,
        lax.max,
        window_dimensions=(1, mp_k, mp_k, mp_k, 1),
        window_strides=(1, mp_k, mp_k, mp_k, 1),
        padding='VALID',
    )
    return jnp.transpose(out, (0, 4, 1, 2, 3))
