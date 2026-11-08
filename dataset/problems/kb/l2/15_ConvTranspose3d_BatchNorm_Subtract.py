"""3D transposed convolution followed by BatchNorm3d and spatial mean subtraction."""

import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': '15_ConvTranspose3d_BatchNorm_Subtract',
    'batch_size': 16,
    'in_channels': 32,
    'out_channels': 32,
    'depth': 32,
    'height': 64,
    'width': 64,
    'kernel_size': 3,
    'stride': 2,
    'padding': 1,
    'eps': 1e-5,
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    k1, k2, k3 = jax.random.split(key, 3)
    B = CONFIG['batch_size']
    C_in = CONFIG['in_channels']
    C_out = CONFIG['out_channels']
    D = CONFIG['depth']
    H = CONFIG['height']
    W = CONFIG['width']
    K = CONFIG['kernel_size']

    x = jax.random.uniform(k1, (B, C_in, D, H, W), dtype=dtype)
    conv_weight = jax.random.normal(k2, (C_in, C_out, K, K, K), dtype=dtype) * 0.02
    conv_bias = jax.random.normal(k3, (C_out,), dtype=dtype) * 0.02
    bn_weight = jnp.ones((C_out,), dtype=dtype)
    bn_bias = jnp.zeros((C_out,), dtype=dtype)
    return x, conv_weight, conv_bias, bn_weight, bn_bias


def workload(x, conv_weight, conv_bias, bn_weight, bn_bias):
    stride = CONFIG['stride']
    pad = CONFIG['padding']
    eps = CONFIG['eps']
    kD, kH, kW = conv_weight.shape[2], conv_weight.shape[3], conv_weight.shape[4]

    x_ndhwc = jnp.transpose(x, (0, 2, 3, 4, 1))
    w_flipped = jnp.flip(conv_weight, axis=(2, 3, 4))
    w_dhwoi = jnp.transpose(w_flipped, (2, 3, 4, 1, 0))

    padding = (
        (kD - 1 - pad, kD - 1 - pad),
        (kH - 1 - pad, kH - 1 - pad),
        (kW - 1 - pad, kW - 1 - pad),
    )
    out_ndhwc = lax.conv_general_dilated(
        lhs=x_ndhwc,
        rhs=w_dhwoi,
        window_strides=(1, 1, 1),
        padding=padding,
        lhs_dilation=(stride, stride, stride),
        dimension_numbers=('NDHWC', 'DHWOI', 'NDHWC'),
    )
    out_ndhwc = out_ndhwc + conv_bias[None, None, None, None, :]
    x = jnp.transpose(out_ndhwc, (0, 4, 1, 2, 3))

    mean = jnp.mean(x, axis=(0, 2, 3, 4), keepdims=True)
    var = jnp.var(x, axis=(0, 2, 3, 4), keepdims=True)
    x = (x - mean) / jnp.sqrt(var + eps) * bn_weight[None, :, None, None, None] + bn_bias[None, :, None, None, None]

    x = x - jnp.mean(x, axis=(2, 3, 4), keepdims=True)
    return x
