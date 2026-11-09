"""Conv2d(1->16,k5,p2) -> ReLU -> MaxPool2 -> Conv2d(16->32,k5,p2) -> ReLU -> MaxPool2 -> flatten -> Linear."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'CnnViewModel_3186',
    'batch': 32768,
    'in_channels': 1,
    'height': 28,
    'width': 28,
    'c1': 16,
    'c2': 32,
    'kernel_size': 5,
    'padding': 2,
    'pool': 2,
    'out_dim': 10,
}


def create_inputs(dtype=jnp.float32):
    c = CONFIG
    k = c['kernel_size']
    keys = jax.random.split(jax.random.key(0), 4)
    x = jax.random.uniform(keys[0], (c['batch'], c['in_channels'], c['height'], c['width']), dtype)
    w1 = jax.random.normal(keys[1], (k, k, c['in_channels'], c['c1']), dtype) * 0.02
    b1 = jnp.zeros((c['c1'],), dtype)
    w2 = jax.random.normal(keys[2], (k, k, c['c1'], c['c2']), dtype) * 0.02
    b2 = jnp.zeros((c['c2'],), dtype)
    feat = c['c2'] * (c['height'] // (c['pool'] ** 2)) * (c['width'] // (c['pool'] ** 2))
    w3 = jax.random.normal(keys[3], (feat, c['out_dim']), dtype) * 0.02
    b3 = jnp.zeros((c['out_dim'],), dtype)
    return x, w1, b1, w2, b2, w3, b3


def _conv_relu(x, w, b):
    p = CONFIG['padding']
    y = lax.conv_general_dilated(
        x, w, window_strides=(1, 1), padding=((p, p), (p, p)),
        dimension_numbers=('NHWC', 'HWIO', 'NHWC'))
    return jax.nn.relu(y + b)


def _maxpool(x):
    p = CONFIG['pool']
    return lax.reduce_window(x, -jnp.inf, lax.max, (1, p, p, 1), (1, p, p, 1), 'VALID')


def workload(x, w1, b1, w2, b2, w3, b3):
    c = CONFIG
    x = x.reshape(-1, c['in_channels'], c['height'], c['width'])
    h = jnp.transpose(x, (0, 2, 3, 1))
    h = _maxpool(_conv_relu(h, w1, b1))
    h = _maxpool(_conv_relu(h, w2, b2))
    # Flatten in NCHW order to match torch's view(B, -1).
    h = jnp.transpose(h, (0, 3, 1, 2)).reshape(h.shape[0], -1)
    return h @ w3 + b3
