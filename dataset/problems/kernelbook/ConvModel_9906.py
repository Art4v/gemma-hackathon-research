"""Conv2d+ReLU+MaxPool -> Conv2d+ReLU+MaxPool -> flatten -> Linear+ReLU -> Linear+ReLU -> Linear (LeNet)."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'ConvModel_9906',
    'batch': 16384,
    'in_channels': 3,
    'height': 32,
    'width': 32,
    'conv1_out': 6,
    'conv2_out': 16,
    'kernel_size': 5,
    'pool_size': 2,
    'pool_stride': 2,
    'fc1_out': 200,
    'fc2_out': 100,
    'fc3_out': 10,
}


def create_inputs(dtype=jnp.float32):
    c = CONFIG
    k = c['kernel_size']
    keys = jax.random.split(jax.random.key(0), 11)
    x = jax.random.uniform(keys[0], (c['batch'], c['in_channels'], c['height'], c['width']), dtype)
    # Spatial size after two (valid conv, pool) stages.
    s = c['height']
    for _ in range(2):
        s = (s - k + 1) // c['pool_stride']
    flat = c['conv2_out'] * s * s
    w1 = jax.random.normal(keys[1], (k, k, c['in_channels'], c['conv1_out']), dtype) * 0.02
    b1 = jax.random.normal(keys[2], (c['conv1_out'],), dtype) * 0.02
    w2 = jax.random.normal(keys[3], (k, k, c['conv1_out'], c['conv2_out']), dtype) * 0.02
    b2 = jax.random.normal(keys[4], (c['conv2_out'],), dtype) * 0.02
    fw1 = jax.random.normal(keys[5], (flat, c['fc1_out']), dtype) * 0.02
    fb1 = jax.random.normal(keys[6], (c['fc1_out'],), dtype) * 0.02
    fw2 = jax.random.normal(keys[7], (c['fc1_out'], c['fc2_out']), dtype) * 0.02
    fb2 = jax.random.normal(keys[8], (c['fc2_out'],), dtype) * 0.02
    fw3 = jax.random.normal(keys[9], (c['fc2_out'], c['fc3_out']), dtype) * 0.02
    fb3 = jax.random.normal(keys[10], (c['fc3_out'],), dtype) * 0.02
    return x, w1, b1, w2, b2, fw1, fb1, fw2, fb2, fw3, fb3


def _conv_relu_pool(x, w, b):
    y = lax.conv_general_dilated(
        x, w, window_strides=(1, 1), padding='VALID',
        dimension_numbers=('NHWC', 'HWIO', 'NHWC'))
    y = jax.nn.relu(y + b)
    p, s = CONFIG['pool_size'], CONFIG['pool_stride']
    return lax.reduce_window(
        y, -jnp.inf, lax.max, (1, p, p, 1), (1, s, s, 1), 'VALID')


def workload(x, w1, b1, w2, b2, fw1, fb1, fw2, fb2, fw3, fb3):
    h = jnp.transpose(x, (0, 2, 3, 1))
    h = _conv_relu_pool(h, w1, b1)
    h = _conv_relu_pool(h, w2, b2)
    # Back to NCHW so the flatten order matches torch's view(-1, C*H*W).
    h = jnp.transpose(h, (0, 3, 1, 2)).reshape(h.shape[0], -1)
    h = jax.nn.relu(h @ fw1 + fb1)
    h = jax.nn.relu(h @ fw2 + fb2)
    return h @ fw3 + fb3
