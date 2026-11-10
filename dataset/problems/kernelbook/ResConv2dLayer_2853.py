"""ZeroPad2d -> SpectralNorm Conv2d (1 power iteration) -> LeakyReLU(0.2) -> residual add."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'ResConv2dLayer_2853',
    'batch': 8,
    'channels': 128,
    'height': 256,
    'width': 256,
    'kernel_size': 3,
    'stride': 1,
    'padding': 1,
    'dilation': 1,
    'negative_slope': 0.2,
    'sn_eps': 1e-12,
}

_HI = lax.Precision.HIGHEST


def _l2normalize(v):
    return v / (jnp.sqrt(jnp.sum(v * v)) + CONFIG['sn_eps'])


def create_inputs(dtype=jnp.float32):
    kx, kw, kb, ku = jax.random.split(jax.random.key(0), 4)
    c, k = CONFIG['channels'], CONFIG['kernel_size']
    x = jax.random.uniform(
        kx, (CONFIG['batch'], c, CONFIG['height'], CONFIG['width']), dtype)
    w = jax.random.normal(kw, (c, c, k, k), dtype) * 0.02
    b = jax.random.normal(kb, (c,), dtype) * 0.02
    u = _l2normalize(jax.random.normal(ku, (c,), dtype))
    return x, w, b, u


def workload(x, w_bar, b, u):
    height = w_bar.shape[0]
    w_mat = w_bar.reshape(height, -1)
    v = _l2normalize(jnp.matmul(w_mat.T, u, precision=_HI))
    u_new = _l2normalize(jnp.matmul(w_mat, v, precision=_HI))
    sigma = jnp.dot(u_new, jnp.matmul(w_mat, v, precision=_HI), precision=_HI)
    w = w_bar / sigma

    p = CONFIG['padding']
    x_nhwc = jnp.transpose(x, (0, 2, 3, 1))
    x_pad = jnp.pad(x_nhwc, ((0, 0), (p, p), (p, p), (0, 0)))
    w_hwio = jnp.transpose(w, (2, 3, 1, 0))
    s, d = CONFIG['stride'], CONFIG['dilation']
    y = lax.conv_general_dilated(
        x_pad, w_hwio, window_strides=(s, s), padding='VALID',
        rhs_dilation=(d, d), dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
        precision=_HI)
    y = y + b
    y = jnp.where(y >= 0, y, CONFIG['negative_slope'] * y)
    y = jnp.transpose(y, (0, 3, 1, 2))
    return y + x
