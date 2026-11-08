"""2D convolution followed by bias addition, scaling, sigmoid activation, and GroupNorm."""

import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': '21_Conv2d_Add_Scale_Sigmoid_GroupNorm',
    'batch_size': 128,
    'in_channels': 8,
    'out_channels': 32,
    'height': 256,
    'width': 256,
    'kernel_size': 3,
    'num_groups': 8,
    'eps': 1e-5,
}


def create_inputs(dtype=jnp.float32):
    k0, k1, k2, k3, k4 = jax.random.split(jax.random.key(0), 5)
    B = CONFIG['batch_size']
    C_in = CONFIG['in_channels']
    C_out = CONFIG['out_channels']
    H = CONFIG['height']
    W = CONFIG['width']
    K = CONFIG['kernel_size']

    x = jax.random.uniform(k0, (B, C_in, H, W), dtype=dtype)
    conv_w = jax.random.normal(k1, (C_out, C_in, K, K), dtype=dtype) * 0.02
    conv_b = jax.random.normal(k2, (C_out,), dtype=dtype) * 0.02
    bias = jax.random.normal(k3, (C_out, 1, 1), dtype=dtype) * 0.02
    scale = jax.random.normal(k4, (C_out, 1, 1), dtype=dtype) * 0.02
    gn_w = jnp.ones((C_out,), dtype=dtype)
    gn_b = jnp.zeros((C_out,), dtype=dtype)
    return (x, conv_w, conv_b, bias, scale, gn_w, gn_b)


def workload(x, conv_w, conv_b, bias, scale, gn_w, gn_b):
    x_nhwc = jnp.transpose(x, (0, 2, 3, 1))
    w_hwio = jnp.transpose(conv_w, (2, 3, 1, 0))
    out_nhwc = lax.conv_general_dilated(
        x_nhwc,
        w_hwio,
        window_strides=(1, 1),
        padding='VALID',
        dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
    )
    out = jnp.transpose(out_nhwc, (0, 3, 1, 2)) + conv_b[:, None, None]
    out = out + bias
    out = out * scale
    out = jax.nn.sigmoid(out)

    B, C, H_out, W_out = out.shape
    G = CONFIG['num_groups']
    out_g = jnp.reshape(out, (B, G, C // G, H_out, W_out))
    mean = jnp.mean(out_g, axis=(2, 3, 4), keepdims=True)
    var = jnp.var(out_g, axis=(2, 3, 4), keepdims=True)
    out_g = (out_g - mean) * lax.rsqrt(var + CONFIG['eps'])
    out = jnp.reshape(out_g, (B, C, H_out, W_out))
    out = out * gn_w[:, None, None] + gn_b[:, None, None]
    return out
