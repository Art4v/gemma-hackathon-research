"""[bf16 variant] FGFunction_868: Conv3d -> LeakyReLU -> GroupNorm(1) -> Conv3d -> LeakyReLU -> GroupNorm(1) on an unbatched (C, D, H, W) volume."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'FGFunction_868',
    'channels': 128,
    # torch GroupNorm on the unbatched conv output reads axis 1 (depth) as channels, so depth must equal channels.
    'depth': 128,
    'height': 64,
    'width': 64,
    'kernel': 3,
    'padding': 1,
    'eps': 1e-3,
    'negative_slope': 0.01,
}


def create_inputs(dtype=jnp.bfloat16):
    c, d, h, w = CONFIG['channels'], CONFIG['depth'], CONFIG['height'], CONFIG['width']
    k = CONFIG['kernel']
    kx, k1, k2 = jax.random.split(jax.random.key(0), 3)
    x = jax.random.uniform(kx, (c, d, h, w), dtype)
    w1 = jax.random.normal(k1, (k, k, k, c, c), dtype) * 0.02
    w2 = jax.random.normal(k2, (k, k, k, c, c), dtype) * 0.02
    g1 = jnp.ones((c,), dtype)
    b1 = jnp.zeros((c,), dtype)
    g2 = jnp.ones((c,), dtype)
    b2 = jnp.zeros((c,), dtype)
    return x, w1, g1, b1, w2, g2, b2


def _conv(x, w):
    p = CONFIG['padding']
    h = jnp.transpose(x, (1, 2, 3, 0))[None]
    y = lax.conv_general_dilated(
        h, w, window_strides=(1, 1, 1), padding=[(p, p)] * 3,
        dimension_numbers=('NDHWC', 'DHWIO', 'NDHWC'),
        precision=lax.Precision.HIGHEST)
    return jnp.transpose(y[0], (3, 0, 1, 2))


def _group_norm(x, gamma, beta):
    # Matches torch GroupNorm(1, C) on a 4-D tensor: axis 0 is the batch, stats over axes 1..3, affine on axis 1.
    mean = jnp.mean(x, axis=(1, 2, 3), keepdims=True)
    var = jnp.mean(jnp.square(x - mean), axis=(1, 2, 3), keepdims=True)
    xn = (x - mean) * lax.rsqrt(var + CONFIG['eps'])
    return xn * gamma[None, :, None, None] + beta[None, :, None, None]


def workload(x, w1, g1, b1, w2, g2, b2):
    slope = CONFIG['negative_slope']
    x = _group_norm(jax.nn.leaky_relu(_conv(x, w1), slope), g1, b1)
    x = _group_norm(jax.nn.leaky_relu(_conv(x, w2), slope), g2, b2)
    return x
