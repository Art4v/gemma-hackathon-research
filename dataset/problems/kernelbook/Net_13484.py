"""[bf16 variant] Conv3x3+ReLU+MaxPool2 x3 -> view(-1, 17*17*64) -> Linear+ReLU -> (Dropout=id) -> Linear+Sigmoid."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'Net_13484',
    'batch': 64,
    'in_channels': 3,
    'height': 288,
    'width': 288,
    'c1': 32,
    'c2': 32,
    'c3': 64,
    'kernel': 3,
    'pool': 2,
    'fc1_in': 17 * 17 * 64,
    'fc_hidden': 64,
    'fc_out': 1,
}


def create_inputs(dtype=jnp.bfloat16):
    c = CONFIG
    k = c['kernel']
    keys = jax.random.split(jax.random.key(0), 11)
    x = jax.random.uniform(keys[0], (c['batch'], c['in_channels'], c['height'], c['width']), dtype)
    w1 = jax.random.normal(keys[1], (k, k, c['in_channels'], c['c1']), dtype) * 0.02
    b1 = jax.random.normal(keys[2], (c['c1'],), dtype) * 0.02
    w2 = jax.random.normal(keys[3], (k, k, c['c1'], c['c2']), dtype) * 0.02
    b2 = jax.random.normal(keys[4], (c['c2'],), dtype) * 0.02
    w3 = jax.random.normal(keys[5], (k, k, c['c2'], c['c3']), dtype) * 0.02
    b3 = jax.random.normal(keys[6], (c['c3'],), dtype) * 0.02
    fw1 = jax.random.normal(keys[7], (c['fc1_in'], c['fc_hidden']), dtype) * 0.02
    fb1 = jax.random.normal(keys[8], (c['fc_hidden'],), dtype) * 0.02
    fw2 = jax.random.normal(keys[9], (c['fc_hidden'], c['fc_out']), dtype) * 0.02
    fb2 = jax.random.normal(keys[10], (c['fc_out'],), dtype) * 0.02
    return (x, w1, b1, w2, b2, w3, b3, fw1, fb1, fw2, fb2)


def _conv_relu_pool(x, w, b):
    p = CONFIG['pool']
    y = lax.conv_general_dilated(
        x, w, window_strides=(1, 1), padding='VALID',
        dimension_numbers=('NHWC', 'HWIO', 'NHWC'))
    y = jax.nn.relu(y + b)
    return lax.reduce_window(
        y, -jnp.inf, lax.max, (1, p, p, 1), (1, p, p, 1), 'VALID')


def workload(x, w1, b1, w2, b2, w3, b3, fw1, fb1, fw2, fb2):
    h = jnp.transpose(x, (0, 2, 3, 1))
    h = _conv_relu_pool(h, w1, b1)
    h = _conv_relu_pool(h, w2, b2)
    h = _conv_relu_pool(h, w3, b3)
    # Back to NCHW so the flatten matches torch's view(-1, fc1_in), which
    # splits each sample into several rows at this spatial size.
    h = jnp.transpose(h, (0, 3, 1, 2)).reshape(-1, CONFIG['fc1_in'])
    h = jax.nn.relu(h @ fw1 + fb1)
    return jax.nn.sigmoid(h @ fw2 + fb2)
