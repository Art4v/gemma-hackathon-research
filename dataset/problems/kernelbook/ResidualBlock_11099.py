"""ResidualBlock: x + conv3x3(prelu(conv3x3(x)))."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'ResidualBlock_11099',
    'batch': 16,
    'channels': 128,
    'height': 128,
    'width': 128,
    'kernel_size': 3,
    'padding': 1,
    'prelu_init': 0.25,
}


def create_inputs(dtype=jnp.float32):
    c = CONFIG
    k = jax.random.split(jax.random.key(0), 5)
    ch, ks = c['channels'], c['kernel_size']
    x = jax.random.uniform(k[0], (c['batch'], ch, c['height'], c['width']), dtype)
    w1 = jax.random.normal(k[1], (ch, ch, ks, ks), dtype) * 0.02
    b1 = jax.random.normal(k[2], (ch,), dtype) * 0.02
    w2 = jax.random.normal(k[3], (ch, ch, ks, ks), dtype) * 0.02
    b2 = jax.random.normal(k[4], (ch,), dtype) * 0.02
    alpha = jnp.full((1,), c['prelu_init'], dtype)
    return x, w1, b1, w2, b2, alpha


def _conv(x_nhwc, w_oihw, b):
    p = CONFIG['padding']
    y = lax.conv_general_dilated(
        x_nhwc, w_oihw, window_strides=(1, 1), padding=((p, p), (p, p)),
        dimension_numbers=('NHWC', 'OIHW', 'NHWC'),
        precision=lax.Precision.HIGHEST)
    return y + b


def workload(x, w1, b1, w2, b2, alpha):
    h = jnp.transpose(x, (0, 2, 3, 1))
    r = _conv(h, w1, b1)
    r = jnp.where(r >= 0, r, alpha[0] * r)
    r = _conv(r, w2, b2)
    return x + jnp.transpose(r, (0, 3, 1, 2))
