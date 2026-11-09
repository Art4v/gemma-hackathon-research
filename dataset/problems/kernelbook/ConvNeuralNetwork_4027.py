"""Conv3x3-ReLU x2 -> MaxPool2x2 -> Conv3x3-ReLU x2 -> flatten -> Linear (ConvNeuralNetwork)."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'ConvNeuralNetwork_4027',
    'batch': 2048,
    'in_channels': 3,
    'height': 64,
    'width': 64,
    'c1': 12,
    'c2': 12,
    'c3': 24,
    'c4': 24,
    'kernel_size': 3,
    'padding': 1,
    'pool': 2,
    'fc_in': 32 * 32 * 24,
    'num_classes': 3,
}


def create_inputs(dtype=jnp.float32):
    c = CONFIG
    k = c['kernel_size']
    keys = jax.random.split(jax.random.key(0), 11)
    x = jax.random.uniform(keys[0], (c['batch'], c['in_channels'], c['height'], c['width']), dtype)
    chans = [c['in_channels'], c['c1'], c['c2'], c['c3'], c['c4']]
    params = []
    for i in range(4):
        w = jax.random.normal(keys[1 + 2 * i], (chans[i + 1], chans[i], k, k), dtype) * 0.02
        b = jax.random.normal(keys[2 + 2 * i], (chans[i + 1],), dtype) * 0.02
        params += [w, b]
    fc_w = jax.random.normal(keys[9], (c['fc_in'], c['num_classes']), dtype) * 0.02
    fc_b = jax.random.normal(keys[10], (c['num_classes'],), dtype) * 0.02
    return (x, *params, fc_w, fc_b)


def _conv_relu(x, w, b):
    p = CONFIG['padding']
    y = lax.conv_general_dilated(
        x, w, window_strides=(1, 1), padding=((p, p), (p, p)),
        dimension_numbers=('NHWC', 'OIHW', 'NHWC'))
    return jax.nn.relu(y + b)


def workload(x, w1, b1, w2, b2, w3, b3, w4, b4, fc_w, fc_b):
    s = CONFIG['pool']
    h = jnp.transpose(x, (0, 2, 3, 1))
    h = _conv_relu(h, w1, b1)
    h = _conv_relu(h, w2, b2)
    h = lax.reduce_window(h, -jnp.inf, lax.max, (1, s, s, 1), (1, s, s, 1), 'VALID')
    h = _conv_relu(h, w3, b3)
    h = _conv_relu(h, w4, b4)
    # Flatten in NCHW order to match torch .view
    h = jnp.transpose(h, (0, 3, 1, 2)).reshape(-1, fc_w.shape[0])
    return h @ fc_w + fc_b
