"""ResBlock: relu(conv1x1) -> relu(conv3x3) -> relu(conv1x1), plus relu(conv1x1) shortcut, summed."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'ResBlock_10456',
    'batch': 16,
    'in_channels': 128,
    'mid_channels': 128,
    'out_channels': 256,
    'height': 128,
    'width': 128,
    'kernel_size': 3,
    'padding': 1,
}


def create_inputs(dtype=jnp.float32):
    c = CONFIG
    k = jax.random.split(jax.random.key(0), 5)
    x = jax.random.uniform(k[0], (c['batch'], c['in_channels'], c['height'], c['width']), dtype)
    ks = c['kernel_size']
    w1a = jax.random.normal(k[1], (1, 1, c['in_channels'], c['mid_channels']), dtype) * 0.02
    w1b = jax.random.normal(k[2], (ks, ks, c['mid_channels'], c['mid_channels']), dtype) * 0.02
    w1c = jax.random.normal(k[3], (1, 1, c['mid_channels'], c['out_channels']), dtype) * 0.02
    w2a = jax.random.normal(k[4], (1, 1, c['in_channels'], c['out_channels']), dtype) * 0.02
    return x, w1a, w1b, w1c, w2a


def _conv_relu(x, w, pad):
    y = lax.conv_general_dilated(
        x, w, window_strides=(1, 1), padding=[(pad, pad), (pad, pad)],
        dimension_numbers=('NHWC', 'HWIO', 'NHWC'))
    return jax.nn.relu(y)


def workload(x, w1a, w1b, w1c, w2a):
    h = jnp.transpose(x, (0, 2, 3, 1))
    out1 = _conv_relu(h, w1a, 0)
    out1 = _conv_relu(out1, w1b, CONFIG['padding'])
    out1 = _conv_relu(out1, w1c, 0)
    out2 = _conv_relu(h, w2a, 0)
    return jnp.transpose(out1 + out2, (0, 3, 1, 2))
