"""[bf16 variant] ConvTranspose2d, BatchNorm2d, Tanh, MaxPool2d, and GroupNorm chain."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': '11_ConvTranspose2d_BatchNorm_Tanh_MaxPool_GroupNorm',
    'batch_size': 512,
    'in_channels': 64,
    'out_channels': 128,
    'height': 32,
    'width': 32,
    'kernel_size': 5,
    'stride': 1,
    'padding': 1,
    'pool_size': 2,
    'pool_stride': 2,
    'num_groups': 8,
    'eps': 1e-5,
}


def create_inputs(dtype=jnp.bfloat16):
    k0, k1, k2 = jax.random.split(jax.random.key(0), 3)
    N = CONFIG['batch_size']
    C_in = CONFIG['in_channels']
    C_out = CONFIG['out_channels']
    H = CONFIG['height']
    W = CONFIG['width']
    K = CONFIG['kernel_size']

    x = jax.random.uniform(k0, (N, C_in, H, W), dtype=dtype)
    conv_w = jax.random.normal(k1, (C_in, C_out, K, K), dtype=dtype) * 0.02
    conv_b = jax.random.normal(k2, (C_out,), dtype=dtype) * 0.02
    bn_w = jnp.ones((C_out,), dtype=dtype)
    bn_b = jnp.zeros((C_out,), dtype=dtype)
    gn_w = jnp.ones((C_out,), dtype=dtype)
    gn_b = jnp.zeros((C_out,), dtype=dtype)
    return x, conv_w, conv_b, bn_w, bn_b, gn_w, gn_b


def workload(x, conv_w, conv_b, bn_w, bn_b, gn_w, gn_b):
    stride = CONFIG['stride']
    padding = CONFIG['padding']
    pool_size = CONFIG['pool_size']
    pool_stride = CONFIG['pool_stride']
    num_groups = CONFIG['num_groups']
    eps = CONFIG['eps']

    kh, kw = conv_w.shape[2], conv_w.shape[3]
    pad_h = kh - 1 - padding
    pad_w = kw - 1 - padding
    w_flipped = jnp.flip(conv_w, axis=(2, 3))

    x_nhwc = jnp.transpose(x, (0, 2, 3, 1))
    x_nhwc = lax.conv_general_dilated(
        x_nhwc,
        w_flipped,
        window_strides=(1, 1),
        padding=[(pad_h, pad_h), (pad_w, pad_w)],
        lhs_dilation=(stride, stride),
        dimension_numbers=('NHWC', 'IOHW', 'NHWC'),
    )
    x_nhwc = x_nhwc + conv_b[None, None, None, :]
    x = jnp.transpose(x_nhwc, (0, 3, 1, 2))

    bn_mean = jnp.mean(x, axis=(0, 2, 3), keepdims=True)
    bn_var = jnp.var(x, axis=(0, 2, 3), keepdims=True)
    x = (x - bn_mean) / jnp.sqrt(bn_var + eps) * bn_w[None, :, None, None] + bn_b[None, :, None, None]

    x = jnp.tanh(x)

    x = lax.reduce_window(
        x,
        -jnp.inf,
        lax.max,
        window_dimensions=(1, 1, pool_size, pool_size),
        window_strides=(1, 1, pool_stride, pool_stride),
        padding='VALID',
    )

    N, C, H, W = x.shape
    x = jnp.reshape(x, (N, num_groups, C // num_groups, H, W))
    gn_mean = jnp.mean(x, axis=(2, 3, 4), keepdims=True)
    gn_var = jnp.var(x, axis=(2, 3, 4), keepdims=True)
    x = (x - gn_mean) / jnp.sqrt(gn_var + eps)
    x = jnp.reshape(x, (N, C, H, W))
    x = x * gn_w[None, :, None, None] + gn_b[None, :, None, None]
    return x
