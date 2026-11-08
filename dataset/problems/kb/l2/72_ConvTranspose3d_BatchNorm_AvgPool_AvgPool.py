"""3D transposed convolution followed by batch normalization and two 3D average pooling layers."""

import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': '72_ConvTranspose3d_BatchNorm_AvgPool_AvgPool',
    'batch_size': 32,
    'in_channels': 64,
    'out_channels': 32,
    'depth': 32,
    'height': 32,
    'width': 32,
    'kernel_size': 3,
    'stride': 2,
    'padding': 1,
    'pool_size': 2,
    'eps': 1e-5,
}


def create_inputs(dtype=jnp.float32):
    k1, k2 = jax.random.split(jax.random.key(0), 2)
    b = CONFIG['batch_size']
    c_in = CONFIG['in_channels']
    c_out = CONFIG['out_channels']
    d = CONFIG['depth']
    h = CONFIG['height']
    w = CONFIG['width']
    k = CONFIG['kernel_size']

    x = jax.random.uniform(k1, (b, c_in, d, h, w), dtype=dtype)
    conv_weight = jax.random.normal(k2, (c_in, c_out, k, k, k), dtype=dtype) * 0.02
    conv_bias = jnp.zeros((c_out,), dtype=dtype)
    bn_weight = jnp.ones((c_out,), dtype=dtype)
    bn_bias = jnp.zeros((c_out,), dtype=dtype)
    return (x, conv_weight, conv_bias, bn_weight, bn_bias)


def workload(x, conv_weight, conv_bias, bn_weight, bn_bias):
    stride = CONFIG['stride']
    padding = CONFIG['padding']
    pool_size = CONFIG['pool_size']
    eps = CONFIG['eps']

    k_d, k_h, k_w = conv_weight.shape[2], conv_weight.shape[3], conv_weight.shape[4]
    pad_d = k_d - 1 - padding
    pad_h = k_h - 1 - padding
    pad_w = k_w - 1 - padding

    x_ndhwc = jnp.transpose(x, (0, 2, 3, 4, 1))
    w_flipped = jnp.flip(conv_weight, axis=(2, 3, 4))
    w_dhwoi = jnp.transpose(w_flipped, (2, 3, 4, 1, 0))

    x_ndhwc = lax.conv_general_dilated(
        x_ndhwc,
        w_dhwoi,
        window_strides=(1, 1, 1),
        padding=((pad_d, pad_d), (pad_h, pad_h), (pad_w, pad_w)),
        lhs_dilation=(stride, stride, stride),
        dimension_numbers=('NDHWC', 'DHWOI', 'NDHWC'),
    )
    x_ndhwc = x_ndhwc + conv_bias.reshape(1, 1, 1, 1, -1)

    mean = jnp.mean(x_ndhwc, axis=(0, 1, 2, 3), keepdims=True)
    var = jnp.var(x_ndhwc, axis=(0, 1, 2, 3), keepdims=True)
    x_ndhwc = (x_ndhwc - mean) / jnp.sqrt(var + eps)
    x_ndhwc = x_ndhwc * bn_weight.reshape(1, 1, 1, 1, -1) + bn_bias.reshape(1, 1, 1, 1, -1)

    win = (1, pool_size, pool_size, pool_size, 1)
    pool_vol = pool_size * pool_size * pool_size
    x_ndhwc = lax.reduce_window(x_ndhwc, jnp.asarray(0.0, dtype=x.dtype), lax.add, win, win, 'VALID') / pool_vol
    x_ndhwc = lax.reduce_window(x_ndhwc, jnp.asarray(0.0, dtype=x.dtype), lax.add, win, win, 'VALID') / pool_vol

    return jnp.transpose(x_ndhwc, (0, 4, 1, 2, 3))
