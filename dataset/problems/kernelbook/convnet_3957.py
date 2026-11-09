"""Conv2d 3x3 (pad 1) -> ReLU -> Conv2d 1x1."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'convnet_3957',
    'batch': 16,
    'in_channel': 4,
    'height': 512,
    'width': 512,
    'hidden_channels': 32,
    'out_channels': 1,
    'kernel1': 3,
    'padding1': 1,
    'kernel2': 1,
    'padding2': 0,
}


def create_inputs(dtype=jnp.float32):
    k = jax.random.split(jax.random.key(0), 5)
    c = CONFIG
    x = jax.random.uniform(k[0], (c['batch'], c['in_channel'], c['height'], c['width']), dtype)
    w1 = jax.random.normal(k[1], (c['kernel1'], c['kernel1'], c['in_channel'], c['hidden_channels']), dtype) * 0.02
    b1 = jax.random.normal(k[2], (c['hidden_channels'],), dtype) * 0.02
    w2 = jax.random.normal(k[3], (c['kernel2'], c['kernel2'], c['hidden_channels'], c['out_channels']), dtype) * 0.02
    b2 = jax.random.normal(k[4], (c['out_channels'],), dtype) * 0.02
    return x, w1, b1, w2, b2


def _conv(x, w, b, pad):
    y = lax.conv_general_dilated(
        x, w, window_strides=(1, 1), padding=((pad, pad), (pad, pad)),
        dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
        precision=lax.Precision.HIGHEST)
    return y + b


def workload(x, w1, b1, w2, b2):
    h = jnp.transpose(x, (0, 2, 3, 1))
    h = jax.nn.relu(_conv(h, w1, b1, CONFIG['padding1']))
    h = _conv(h, w2, b2, CONFIG['padding2'])
    return jnp.transpose(h, (0, 3, 1, 2))
