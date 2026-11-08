"""Transposed 2D convolution followed by add, minimum with zero, exact GELU, and multiply."""

import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': '93_ConvTranspose2d_Add_Min_GELU_Multiply',
    'batch_size': 128,
    'in_channels': 128,
    'out_channels': 128,
    'height': 128,
    'width': 128,
    'kernel_size': 4,
    'stride': 2,
    'add_value': 0.5,
    'multiply_value': 2.0,
}


def create_inputs(dtype=jnp.float32):
    k1, k2, k3 = jax.random.split(jax.random.key(0), 3)
    x = jax.random.uniform(
        k1,
        (CONFIG['batch_size'], CONFIG['in_channels'], CONFIG['height'], CONFIG['width']),
        dtype=dtype,
    )
    weight = (
        jax.random.normal(
            k2,
            (CONFIG['in_channels'], CONFIG['out_channels'], CONFIG['kernel_size'], CONFIG['kernel_size']),
            dtype=dtype,
        )
        * 0.02
    )
    bias = jax.random.normal(k3, (CONFIG['out_channels'],), dtype=dtype) * 0.02
    return x, weight, bias


def workload(x, weight, bias):
    stride = CONFIG['stride']
    kh, kw = weight.shape[2], weight.shape[3]

    x_nhwc = jnp.transpose(x, (0, 2, 3, 1))
    w_flipped = jnp.flip(weight, axis=(2, 3))
    w_hwio = jnp.transpose(w_flipped, (2, 3, 0, 1))

    out_nhwc = lax.conv_general_dilated(
        lhs=x_nhwc,
        rhs=w_hwio,
        window_strides=(1, 1),
        padding=((kh - 1, kh - 1), (kw - 1, kw - 1)),
        lhs_dilation=(stride, stride),
        dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
    )
    out = jnp.transpose(out_nhwc, (0, 3, 1, 2))
    out = out + bias.reshape(1, -1, 1, 1)

    out = out + jnp.asarray(CONFIG['add_value'], dtype=out.dtype)
    out = jnp.minimum(out, jnp.asarray(0.0, dtype=out.dtype))
    inv_sqrt2 = jnp.asarray(0.7071067811865475, dtype=out.dtype)
    out = jnp.asarray(0.5, dtype=out.dtype) * out * (jnp.asarray(1.0, dtype=out.dtype) + lax.erf(out * inv_sqrt2))
    out = out * jnp.asarray(CONFIG['multiply_value'], dtype=out.dtype)
    return out
