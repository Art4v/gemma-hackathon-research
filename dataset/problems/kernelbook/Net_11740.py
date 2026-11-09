"""[bf16 variant] Conv5x5+ReLU+MaxPool2 -> Conv3x3+ReLU+MaxPool3 -> view(-1, 41472) -> Linear+ReLU -> (Dropout=id) -> Linear."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'Net_11740',
    'batch': 1024,
    'in_channels': 1,
    'height': 121,
    'width': 121,
    'conv1_out': 32,
    'conv1_k': 5,
    'pool1': 2,
    'conv2_out': 32,
    'conv2_k': 3,
    'pool2': 3,
    'fc1_in': 36 * 36 * 32,
    'hidden': 64,
    'out_features': 136,
}


def create_inputs(dtype=jnp.bfloat16):
    c = CONFIG
    keys = jax.random.split(jax.random.key(0), 5)
    x = jax.random.uniform(keys[0], (c['batch'], c['in_channels'], c['height'], c['width']), dtype)
    w1 = jax.random.normal(keys[1], (c['conv1_out'], c['in_channels'], c['conv1_k'], c['conv1_k']), dtype) * 0.02
    b1 = jnp.zeros((c['conv1_out'],), dtype)
    w2 = jax.random.normal(keys[2], (c['conv2_out'], c['conv1_out'], c['conv2_k'], c['conv2_k']), dtype) * 0.02
    b2 = jnp.zeros((c['conv2_out'],), dtype)
    fw1 = jax.random.normal(keys[3], (c['fc1_in'], c['hidden']), dtype) * 0.02
    fb1 = jnp.zeros((c['hidden'],), dtype)
    fw2 = jax.random.normal(keys[4], (c['hidden'], c['out_features']), dtype) * 0.02
    fb2 = jnp.zeros((c['out_features'],), dtype)
    return x, w1, b1, w2, b2, fw1, fb1, fw2, fb2


def _conv_relu(x, w, b):
    # x NHWC, w OIHW (PyTorch layout), VALID padding, stride 1.
    y = lax.conv_general_dilated(x, w, (1, 1), 'VALID', dimension_numbers=('NHWC', 'OIHW', 'NHWC'))
    return jax.nn.relu(y + b)


def _maxpool(x, k):
    # VALID padding == PyTorch floor mode.
    return lax.reduce_window(x, -jnp.inf, lax.max, (1, k, k, 1), (1, k, k, 1), 'VALID')


def workload(x, w1, b1, w2, b2, fw1, fb1, fw2, fb2):
    h = jnp.transpose(x, (0, 2, 3, 1))
    h = _maxpool(_conv_relu(h, w1, b1), CONFIG['pool1'])
    h = _maxpool(_conv_relu(h, w2, b2), CONFIG['pool2'])
    # PyTorch view flattens NCHW and packs several images per row (18*18*32 per image vs fc1_in).
    h = jnp.transpose(h, (0, 3, 1, 2)).reshape(-1, CONFIG['fc1_in'])
    h = jax.nn.relu(h @ fw1 + fb1)
    return h @ fw2 + fb2
