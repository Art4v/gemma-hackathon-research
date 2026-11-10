"""ResidualBlock: mfm(conv3x3 -> split -> max) x2 + residual add."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'ResidualBlock_4932',
    'batch': 8,
    'channels': 128,
    'height': 128,
    'width': 128,
    'kernel_size': 3,
    'stride': 1,
    'padding': 1,
}


def create_inputs(dtype=jnp.float32):
    k = jax.random.key(0)
    kx, kw1, kb1, kw2, kb2 = jax.random.split(k, 5)
    n, c, h, w = CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width']
    ks = CONFIG['kernel_size']
    x = jax.random.uniform(kx, (n, c, h, w), dtype)
    w1 = jax.random.normal(kw1, (ks, ks, c, 2 * c), dtype) * 0.02
    b1 = jax.random.normal(kb1, (2 * c,), dtype) * 0.02
    w2 = jax.random.normal(kw2, (ks, ks, c, 2 * c), dtype) * 0.02
    b2 = jax.random.normal(kb2, (2 * c,), dtype) * 0.02
    return x, w1, b1, w2, b2


def _mfm(x_nhwc, w_hwio, b):
    s, p = CONFIG['stride'], CONFIG['padding']
    y = lax.conv_general_dilated(
        x_nhwc, w_hwio, window_strides=(s, s), padding=((p, p), (p, p)),
        dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
        precision=lax.Precision.HIGHEST,
    ) + b
    c = w_hwio.shape[-1] // 2
    return jnp.maximum(y[..., :c], y[..., c:])


def workload(x, w1, b1, w2, b2):
    h = jnp.transpose(x, (0, 2, 3, 1))
    out = _mfm(h, w1, b1)
    out = _mfm(out, w2, b2)
    out = out + h
    return jnp.transpose(out, (0, 3, 1, 2))
