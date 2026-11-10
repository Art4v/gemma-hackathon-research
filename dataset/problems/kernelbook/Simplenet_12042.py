"""Simplenet: conv5x5 -> ReLU -> maxpool2 -> conv5x5 -> ReLU -> maxpool2 -> flatten -> linear -> ReLU -> linear."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'Simplenet_12042',
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
    'fc1_out': 120,
    'fc2_out': 100,
}


def create_inputs(dtype=jnp.float32):
    c = CONFIG
    keys = jax.random.split(jax.random.key(0), 9)
    k = c['kernel_size']
    x = jax.random.uniform(keys[0], (c['batch'], c['in_channels'], c['height'], c['width']), dtype)
    w1 = jax.random.normal(keys[1], (k, k, c['in_channels'], c['conv1_out']), dtype) * 0.02
    b1 = jax.random.normal(keys[2], (c['conv1_out'],), dtype) * 0.02
    w2 = jax.random.normal(keys[3], (k, k, c['conv1_out'], c['conv2_out']), dtype) * 0.02
    b2 = jax.random.normal(keys[4], (c['conv2_out'],), dtype) * 0.02
    fw1 = jax.random.normal(keys[5], (c['flat_features'], c['fc1_out']), dtype) * 0.02
    fb1 = jax.random.normal(keys[6], (c['fc1_out'],), dtype) * 0.02
    fw2 = jax.random.normal(keys[7], (c['fc1_out'], c['fc2_out']), dtype) * 0.02
    fb2 = jax.random.normal(keys[8], (c['fc2_out'],), dtype) * 0.02
    return (x, w1, b1, w2, b2, fw1, fb1, fw2, fb2)


def _conv_relu_pool(x, w, b):
    # x: NHWC, w: HWIO. PyTorch Conv2d defaults: stride 1, no padding.
    y = lax.conv_general_dilated(
        x, w, window_strides=(1, 1), padding='VALID',
        dimension_numbers=('NHWC', 'HWIO', 'NHWC'))
    y = jax.nn.relu(y + b)
    p, s = CONFIG['pool_size'], CONFIG['pool_stride']
    return lax.reduce_window(y, -jnp.inf, lax.max, (1, p, p, 1), (1, s, s, 1), 'VALID')


def workload(x, w1, b1, w2, b2, fw1, fb1, fw2, fb2):
    h = jnp.transpose(x, (0, 2, 3, 1))
    h = _conv_relu_pool(h, w1, b1)
    h = _conv_relu_pool(h, w2, b2)
    # Flatten in NCHW order to match torch's x.view(-1, 16*5*5).
    h = jnp.transpose(h, (0, 3, 1, 2)).reshape(-1, CONFIG['flat_features'])
    h = jax.nn.relu(h @ fw1 + fb1)
    return h @ fw2 + fb2
