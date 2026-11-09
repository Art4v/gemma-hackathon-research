"""DoubleConv: conv3x3 (valid) + bias -> ReLU -> conv3x3 (valid) + bias -> ReLU."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'DoubleConv_8847',
    'batch': 8,
    'in_channels': 128,
    'out_channels': 128,
    'height': 256,
    'width': 256,
    'kernel_size': 3,
    'stride': 1,
    'padding': 'VALID',
}


def create_inputs(dtype=jnp.float32):
    k = jax.random.split(jax.random.key(0), 5)
    ks = CONFIG['kernel_size']
    cin, cout = CONFIG['in_channels'], CONFIG['out_channels']
    x = jax.random.uniform(
        k[0], (CONFIG['batch'], cin, CONFIG['height'], CONFIG['width']), dtype)
    w1 = jax.random.normal(k[1], (ks, ks, cin, cout), dtype) * 0.02
    b1 = jax.random.normal(k[2], (cout,), dtype) * 0.02
    w2 = jax.random.normal(k[3], (ks, ks, cout, cout), dtype) * 0.02
    b2 = jax.random.normal(k[4], (cout,), dtype) * 0.02
    return x, w1, b1, w2, b2


def _conv_relu(x, w, b):
    s = CONFIG['stride']
    y = lax.conv_general_dilated(
        x, w, window_strides=(s, s), padding=CONFIG['padding'],
        dimension_numbers=('NHWC', 'HWIO', 'NHWC'))
    return jax.nn.relu(y + b)


def workload(x, w1, b1, w2, b2):
    h = jnp.transpose(x, (0, 2, 3, 1))
    h = _conv_relu(h, w1, b1)
    h = _conv_relu(h, w2, b2)
    return jnp.transpose(h, (0, 3, 1, 2))
