"""3D transposed convolution, scalar multiplication, 3D max pooling, global average pooling, and clamping."""

import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': '96_ConvTranspose3d_Multiply_Max_GlobalAvgPool_Clamp',
    'batch_size': 16,
    'in_channels': 32,
    'out_channels': 64,
    'depth': 32,
    'height': 64,
    'width': 64,
    'kernel_size': 3,
    'stride': 2,
    'padding': 1,
    'scale': 0.5,
    'maxpool_kernel_size': 2,
    'clamp_min': 0.0,
    'clamp_max': 1.0,
}


def create_inputs(dtype=jnp.float32):
    k1, k2, k3 = jax.random.split(jax.random.key(0), 3)
    b = CONFIG['batch_size']
    c_in = CONFIG['in_channels']
    c_out = CONFIG['out_channels']
    d = CONFIG['depth']
    h = CONFIG['height']
    w = CONFIG['width']
    k = CONFIG['kernel_size']

    x = jax.random.uniform(k1, (b, c_in, d, h, w), dtype=dtype)
    weight = jax.random.normal(k2, (c_in, c_out, k, k, k), dtype=dtype) * 0.02
    bias = jax.random.normal(k3, (c_out,), dtype=dtype) * 0.02
    return x, weight, bias


def workload(x, weight, bias):
    s = CONFIG['stride']
    p = CONFIG['padding']
    k = weight.shape[2]
    mp = CONFIG['maxpool_kernel_size']

    x_ndhwc = jnp.transpose(x, (0, 2, 3, 4, 1))
    w_flipped = jnp.flip(weight, axis=(2, 3, 4))
    pad = k - 1 - p
    padding = ((pad, pad), (pad, pad), (pad, pad))

    out = lax.conv_general_dilated(
        lhs=x_ndhwc,
        rhs=w_flipped,
        window_strides=(1, 1, 1),
        padding=padding,
        lhs_dilation=(s, s, s),
        dimension_numbers=('NDHWC', 'IODHW', 'NDHWC'),
    )
    out = out + jnp.reshape(bias, (1, 1, 1, 1, -1))
    out = out * CONFIG['scale']

    out = lax.reduce_window(
        out,
        -jnp.inf,
        lax.max,
        window_dimensions=(1, mp, mp, mp, 1),
        window_strides=(1, mp, mp, mp, 1),
        padding='VALID',
    )

    out = jnp.mean(out, axis=(1, 2, 3), keepdims=True)
    out = jnp.clip(out, CONFIG['clamp_min'], CONFIG['clamp_max'])
    out = jnp.transpose(out, (0, 4, 1, 2, 3))
    return out
