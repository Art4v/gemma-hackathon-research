"""Conv2d -> ReLU -> MaxPool -> Conv2d -> ReLU -> MaxPool -> Flatten -> Linear -> ReLU -> Linear -> ReLU -> Linear."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'Net_531',
    'batch': 8192,
    'in_channels': 3,
    'height': 32,
    'width': 32,
    'conv1_out': 100,
    'conv2_out': 16,
    'kernel_size': 5,
    'pool': 2,
    'fc1_out': 120,
    'fc2_out': 84,
    'fc3_out': 10,
}


def create_inputs(dtype=jnp.float32):
    c = CONFIG
    k = c['kernel_size']
    keys = jax.random.split(jax.random.key(0), 11)
    x = jax.random.uniform(keys[0], (c['batch'], c['in_channels'], c['height'], c['width']), dtype)
    # Spatial size after two (valid conv, pool) stages.
    s = ((c['height'] - k + 1) // c['pool'] - k + 1) // c['pool']
    flat = c['conv2_out'] * s * s

    def w(key, shape):
        return jax.random.normal(key, shape, dtype) * 0.02

    conv1_w = w(keys[1], (c['conv1_out'], c['in_channels'], k, k))
    conv1_b = w(keys[2], (c['conv1_out'],))
    conv2_w = w(keys[3], (c['conv2_out'], c['conv1_out'], k, k))
    conv2_b = w(keys[4], (c['conv2_out'],))
    fc1_w = w(keys[5], (flat, c['fc1_out']))
    fc1_b = w(keys[6], (c['fc1_out'],))
    fc2_w = w(keys[7], (c['fc1_out'], c['fc2_out']))
    fc2_b = w(keys[8], (c['fc2_out'],))
    fc3_w = w(keys[9], (c['fc2_out'], c['fc3_out']))
    fc3_b = w(keys[10], (c['fc3_out'],))
    return (x, conv1_w, conv1_b, conv2_w, conv2_b, fc1_w, fc1_b, fc2_w, fc2_b, fc3_w, fc3_b)


def _conv_relu_pool(x, w, b):
    y = lax.conv_general_dilated(
        x, w, window_strides=(1, 1), padding='VALID',
        dimension_numbers=('NHWC', 'OIHW', 'NHWC'))
    y = jax.nn.relu(y + b)
    p = CONFIG['pool']
    return lax.reduce_window(y, -jnp.inf, lax.max, (1, p, p, 1), (1, p, p, 1), 'VALID')


def workload(x, conv1_w, conv1_b, conv2_w, conv2_b, fc1_w, fc1_b, fc2_w, fc2_b, fc3_w, fc3_b):
    h = jnp.transpose(x, (0, 2, 3, 1))
    h = _conv_relu_pool(h, conv1_w, conv1_b)
    h = _conv_relu_pool(h, conv2_w, conv2_b)
    # Flatten in PyTorch's (C, H, W) order.
    h = jnp.transpose(h, (0, 3, 1, 2)).reshape(h.shape[0], -1)
    h = jax.nn.relu(h @ fc1_w + fc1_b)
    h = jax.nn.relu(h @ fc2_w + fc2_b)
    return h @ fc3_w + fc3_b
