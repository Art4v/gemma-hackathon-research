"""ConvBlock_3007: Conv2d -> ReLU -> MaxPool2d -> Conv2d -> ReLU -> MaxPool2d -> flatten."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'ConvBlock_3007',
    'batch': 8192,
    'in_channels': 3,
    'height': 32,
    'width': 32,
    'conv1_out': 6,
    'conv2_out': 16,
    'kernel_size': 5,
    'pool_size': 2,
    'pool_stride': 2,
    'flat_features': 16 * 5 * 5,
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    kx, k1, kb1, k2, kb2 = jax.random.split(key, 5)
    c = CONFIG
    k = c['kernel_size']
    x = jax.random.uniform(kx, (c['batch'], c['in_channels'], c['height'], c['width']), dtype)
    w1 = jax.random.normal(k1, (k, k, c['in_channels'], c['conv1_out']), dtype) * 0.02
    b1 = jax.random.normal(kb1, (c['conv1_out'],), dtype) * 0.02
    w2 = jax.random.normal(k2, (k, k, c['conv1_out'], c['conv2_out']), dtype) * 0.02
    b2 = jax.random.normal(kb2, (c['conv2_out'],), dtype) * 0.02
    return x, w1, b1, w2, b2


def _conv_relu(x, w, b):
    y = lax.conv_general_dilated(
        x, w, window_strides=(1, 1), padding='VALID',
        dimension_numbers=('NHWC', 'HWIO', 'NHWC'))
    return jax.nn.relu(y + b)


def _maxpool(x):
    p, s = CONFIG['pool_size'], CONFIG['pool_stride']
    return lax.reduce_window(x, -jnp.inf, lax.max, (1, p, p, 1), (1, s, s, 1), 'VALID')


def workload(x, w1, b1, w2, b2):
    h = jnp.transpose(x, (0, 2, 3, 1))
    h = _maxpool(_conv_relu(h, w1, b1))
    h = _maxpool(_conv_relu(h, w2, b2))
    # Back to NCHW so the flatten order matches torch's x.view(-1, 16*5*5).
    h = jnp.transpose(h, (0, 3, 1, 2))
    return h.reshape(-1, CONFIG['flat_features'])
