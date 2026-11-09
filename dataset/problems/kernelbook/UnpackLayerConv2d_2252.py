"""Pad -> Conv2d(k=4, s=1) -> GroupNorm(16) -> ELU -> PixelShuffle(2)."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'UnpackLayerConv2d_2252',
    'batch': 16,
    'in_channels': 128,
    'out_channels': 32,
    'height': 128,
    'width': 128,
    'kernel_size': 4,
    'r': 2,
    'num_groups': 16,
    'eps': 1e-5,
}


def create_inputs(dtype=jnp.float32):
    c = CONFIG
    k_x, k_w, k_b = jax.random.split(jax.random.key(0), 3)
    conv_out = c['out_channels'] * c['r'] ** 2
    ks = c['kernel_size']
    x = jax.random.uniform(k_x, (c['batch'], c['in_channels'], c['height'], c['width']), dtype)
    w = jax.random.normal(k_w, (ks, ks, c['in_channels'], conv_out), dtype) * 0.02
    b = jax.random.normal(k_b, (conv_out,), dtype) * 0.02
    gamma = jnp.ones((conv_out,), dtype)
    beta = jnp.zeros((conv_out,), dtype)
    return x, w, b, gamma, beta


def workload(x, w, b, gamma, beta):
    r = CONFIG['r']
    groups = CONFIG['num_groups']
    eps = CONFIG['eps']
    pad = w.shape[0] // 2
    xh = jnp.transpose(x, (0, 2, 3, 1))
    xh = jnp.pad(xh, ((0, 0), (pad, pad), (pad, pad), (0, 0)))
    y = lax.conv_general_dilated(
        xh, w, window_strides=(1, 1), padding='VALID',
        dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
        precision=lax.Precision.HIGHEST,
    ) + b
    bsz, h, wd, ch = y.shape
    yg = y.reshape(bsz, h, wd, groups, ch // groups)
    mean = jnp.mean(yg, axis=(1, 2, 4), keepdims=True)
    var = jnp.mean(jnp.square(yg - mean), axis=(1, 2, 4), keepdims=True)
    y = ((yg - mean) * lax.rsqrt(var + eps)).reshape(bsz, h, wd, ch) * gamma + beta
    y = jax.nn.elu(y)
    c_out = ch // (r * r)
    y = y.reshape(bsz, h, wd, c_out, r, r)
    y = jnp.transpose(y, (0, 3, 1, 4, 2, 5))
    return y.reshape(bsz, c_out, h * r, wd * r)
