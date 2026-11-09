"""[bf16 variant] Conv3x3+ReLU+MaxPool x3 -> view(-1, 288) -> Linear+ReLU -> Linear+ReLU -> Linear."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'Baseline_9225',
    'batch': 2048,
    'in_channels': 3,
    'height': 64,
    'width': 64,
    'c1': 6,
    'c2': 16,
    'c3': 32,
    'kernel': 3,
    'pool': 2,
    'flat_dim': 32 * 3 * 3,
    'fc1': 120,
    'fc2': 84,
    'fc3': 10,
}


def create_inputs(dtype=jnp.bfloat16):
    c = CONFIG
    k = c['kernel']
    keys = jax.random.split(jax.random.key(0), 13)
    x = jax.random.uniform(keys[0], (c['batch'], c['in_channels'], c['height'], c['width']), dtype)

    def w(i, shape):
        return jax.random.normal(keys[i], shape, dtype) * 0.02

    w1 = w(1, (c['c1'], c['in_channels'], k, k))
    b1 = w(2, (c['c1'],))
    w2 = w(3, (c['c2'], c['c1'], k, k))
    b2 = w(4, (c['c2'],))
    w3 = w(5, (c['c3'], c['c2'], k, k))
    b3 = w(6, (c['c3'],))
    fw1 = w(7, (c['flat_dim'], c['fc1']))
    fb1 = w(8, (c['fc1'],))
    fw2 = w(9, (c['fc1'], c['fc2']))
    fb2 = w(10, (c['fc2'],))
    fw3 = w(11, (c['fc2'], c['fc3']))
    fb3 = w(12, (c['fc3'],))
    return (x, w1, b1, w2, b2, w3, b3, fw1, fb1, fw2, fb2, fw3, fb3)


def _conv_relu_pool(x, w, b):
    y = lax.conv_general_dilated(
        x, w, window_strides=(1, 1), padding='VALID',
        dimension_numbers=('NHWC', 'OIHW', 'NHWC'))
    y = jax.nn.relu(y + b)
    p = CONFIG['pool']
    return lax.reduce_window(y, -jnp.inf, lax.max, (1, p, p, 1), (1, p, p, 1), 'VALID')


def workload(x, w1, b1, w2, b2, w3, b3, fw1, fb1, fw2, fb2, fw3, fb3):
    h = jnp.transpose(x, (0, 2, 3, 1))
    h = _conv_relu_pool(h, w1, b1)
    h = _conv_relu_pool(h, w2, b2)
    h = _conv_relu_pool(h, w3, b3)
    # Match torch .view(-1, 288) on an NCHW tensor: flatten in C,H,W order.
    h = jnp.transpose(h, (0, 3, 1, 2)).reshape(-1, CONFIG['flat_dim'])
    h = jax.nn.relu(h @ fw1 + fb1)
    h = jax.nn.relu(h @ fw2 + fb2)
    return h @ fw3 + fb3
