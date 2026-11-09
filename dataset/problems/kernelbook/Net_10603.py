"""Conv5x5-ReLU-MaxPool x2 -> flatten -> Linear-ReLU-Linear-ReLU-Linear (LeNet)."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'Net_10603',
    'batch': 8192,
    'in_channels': 3,
    'height': 32,
    'width': 32,
    'conv1_out': 12,
    'conv2_out': 16,
    'kernel_size': 5,
    'pool': 2,
    'fc1_out': 120,
    'fc2_out': 84,
    'num_classes': 10,
}


def create_inputs(dtype=jnp.float32):
    c = CONFIG
    k = c['kernel_size']
    flat = c['conv2_out'] * 5 * 5
    keys = jax.random.split(jax.random.key(0), 11)
    x = jax.random.uniform(keys[0], (c['batch'], c['in_channels'], c['height'], c['width']), dtype)
    w1 = jax.random.normal(keys[1], (c['conv1_out'], c['in_channels'], k, k), dtype) * 0.02
    b1 = jax.random.normal(keys[2], (c['conv1_out'],), dtype) * 0.02
    w2 = jax.random.normal(keys[3], (c['conv2_out'], c['conv1_out'], k, k), dtype) * 0.02
    b2 = jax.random.normal(keys[4], (c['conv2_out'],), dtype) * 0.02
    fw1 = jax.random.normal(keys[5], (flat, c['fc1_out']), dtype) * 0.02
    fb1 = jax.random.normal(keys[6], (c['fc1_out'],), dtype) * 0.02
    fw2 = jax.random.normal(keys[7], (c['fc1_out'], c['fc2_out']), dtype) * 0.02
    fb2 = jax.random.normal(keys[8], (c['fc2_out'],), dtype) * 0.02
    fw3 = jax.random.normal(keys[9], (c['fc2_out'], c['num_classes']), dtype) * 0.02
    fb3 = jax.random.normal(keys[10], (c['num_classes'],), dtype) * 0.02
    return x, w1, b1, w2, b2, fw1, fb1, fw2, fb2, fw3, fb3


def _conv_relu_pool(x, w, b):
    p = CONFIG['pool']
    y = lax.conv_general_dilated(
        x, w, window_strides=(1, 1), padding='VALID',
        dimension_numbers=('NHWC', 'OIHW', 'NHWC'))
    y = jax.nn.relu(y + b)
    return lax.reduce_window(y, -jnp.inf, lax.max, (1, p, p, 1), (1, p, p, 1), 'VALID')


def workload(x, w1, b1, w2, b2, fw1, fb1, fw2, fb2, fw3, fb3):
    h = jnp.transpose(x, (0, 2, 3, 1))
    h = _conv_relu_pool(h, w1, b1)
    h = _conv_relu_pool(h, w2, b2)
    # Flatten in NCHW order to match PyTorch's view(-1, C*H*W).
    h = jnp.transpose(h, (0, 3, 1, 2)).reshape(h.shape[0], -1)
    h = jax.nn.relu(h @ fw1 + fb1)
    h = jax.nn.relu(h @ fw2 + fb2)
    return h @ fw3 + fb3
