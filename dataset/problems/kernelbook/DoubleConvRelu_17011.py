"""Conv2d(3x3, pad 1) -> ReLU -> Conv2d(3x3, pad 1) -> ReLU."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'DoubleConvRelu_17011',
    'batch': 16,
    'in_dec_filters': 64,
    'out_filters': 128,
    'height': 128,
    'width': 128,
    'kernel_size': 3,
    'padding': 1,
    'stride': 1,
}


def create_inputs(dtype=jnp.float32):
    k = CONFIG['kernel_size']
    cin, cout = CONFIG['in_dec_filters'], CONFIG['out_filters']
    kx, k1, kb1, k2, kb2 = jax.random.split(jax.random.key(0), 5)
    x = jax.random.uniform(kx, (CONFIG['batch'], cin, CONFIG['height'], CONFIG['width']), dtype)
    w1 = jax.random.normal(k1, (k, k, cin, cout), dtype) * 0.02
    b1 = jax.random.normal(kb1, (cout,), dtype) * 0.02
    w2 = jax.random.normal(k2, (k, k, cout, cout), dtype) * 0.02
    b2 = jax.random.normal(kb2, (cout,), dtype) * 0.02
    return x, w1, b1, w2, b2


def _conv_relu(x, w, b):
    s, p = CONFIG['stride'], CONFIG['padding']
    y = lax.conv_general_dilated(
        x, w, window_strides=(s, s), padding=((p, p), (p, p)),
        dimension_numbers=('NHWC', 'HWIO', 'NHWC'))
    return jax.nn.relu(y + b)


def workload(x, w1, b1, w2, b2):
    h = jnp.transpose(x, (0, 2, 3, 1))
    h = _conv_relu(h, w1, b1)
    h = _conv_relu(h, w2, b2)
    return jnp.transpose(h, (0, 3, 1, 2))
