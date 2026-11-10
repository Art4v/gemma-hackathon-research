"""Conv2d(3x3, pad=1) -> PReLU -> Conv2d(3x3, pad=1) -> PReLU (OutputBlock)."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'OutputBlock_9649',
    'batch': 8,
    'in_channels': 128,
    'out_channels': 128,
    'height': 128,
    'width': 128,
    'kernel_size': 3,
    'padding': 1,
    'prelu_init': 0.25,
}


def create_inputs(dtype=jnp.float32):
    c = CONFIG
    k = c['kernel_size']
    cin, cout = c['in_channels'], c['out_channels']
    kx, kw1, kb1, kw2, kb2 = jax.random.split(jax.random.key(0), 5)
    x = jax.random.uniform(kx, (c['batch'], cin, c['height'], c['width']), dtype)
    w1 = jax.random.normal(kw1, (k, k, cin, cin), dtype) * 0.02
    b1 = jax.random.normal(kb1, (cin,), dtype) * 0.02
    a1 = jnp.full((cin,), c['prelu_init'], dtype)
    w2 = jax.random.normal(kw2, (k, k, cin, cout), dtype) * 0.02
    b2 = jax.random.normal(kb2, (cout,), dtype) * 0.02
    a2 = jnp.full((cout,), c['prelu_init'], dtype)
    return x, w1, b1, a1, w2, b2, a2


def _conv(x, w, b):
    p = CONFIG['padding']
    y = lax.conv_general_dilated(
        x, w, window_strides=(1, 1), padding=((p, p), (p, p)),
        dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
        precision=lax.Precision.HIGHEST)
    return y + b


def _prelu(x, a):
    return jnp.where(x >= 0, x, a * x)


def workload(x, w1, b1, a1, w2, b2, a2):
    h = jnp.transpose(x, (0, 2, 3, 1))
    h = _prelu(_conv(h, w1, b1), a1)
    h = _prelu(_conv(h, w2, b2), a2)
    return jnp.transpose(h, (0, 3, 1, 2))
