"""3D transposed convolution, scalar scaling, 3D batch normalization, and global average pooling."""

import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': '77_ConvTranspose3d_Scale_BatchNorm_GlobalAvgPool',
    'batch_size': 16,
    'in_channels': 128,
    'out_channels': 128,
    'depth': 16,
    'height': 32,
    'width': 32,
    'kernel_size': 5,
    'scale_factor': 2.0,
    'eps': 1e-5,
}


def create_inputs(dtype=jnp.float32):
    k1, k2, k3 = jax.random.split(jax.random.key(0), 3)
    b = CONFIG['batch_size']
    c_in = CONFIG['in_channels']
    c_out = CONFIG['out_channels']
    d, h, w = CONFIG['depth'], CONFIG['height'], CONFIG['width']
    k = CONFIG['kernel_size']

    x = jax.random.uniform(k1, (b, c_in, d, h, w), dtype=dtype)
    weight = jax.random.normal(k2, (c_in, c_out, k, k, k), dtype=dtype) * 0.02
    bias = jax.random.normal(k3, (c_out,), dtype=dtype) * 0.02
    bn_weight = jnp.ones((c_out,), dtype=dtype)
    bn_bias = jnp.zeros((c_out,), dtype=dtype)
    return (x, weight, bias, bn_weight, bn_bias)


def workload(x, weight, bias, bn_weight, bn_bias):
    kd, kh, kw = weight.shape[2], weight.shape[3], weight.shape[4]
    pad = [(kd - 1, kd - 1), (kh - 1, kh - 1), (kw - 1, kw - 1)]

    x_ndhwc = jnp.transpose(x, (0, 2, 3, 4, 1))
    w_dhwoi = jnp.transpose(jnp.flip(weight, axis=(2, 3, 4)), (2, 3, 4, 1, 0))
    out = lax.conv_general_dilated(
        x_ndhwc,
        w_dhwoi,
        window_strides=(1, 1, 1),
        padding=pad,
        dimension_numbers=('NDHWC', 'DHWOI', 'NDHWC'),
    )
    out = out + bias
    x = jnp.transpose(out, (0, 4, 1, 2, 3))

    x = x * CONFIG['scale_factor']

    mean = jnp.mean(x, axis=(0, 2, 3, 4), keepdims=True)
    var = jnp.var(x, axis=(0, 2, 3, 4), keepdims=True)
    x = (x - mean) / jnp.sqrt(var + CONFIG['eps'])
    x = x * bn_weight[None, :, None, None, None] + bn_bias[None, :, None, None, None]

    x = jnp.mean(x, axis=(2, 3, 4), keepdims=True)
    return x
