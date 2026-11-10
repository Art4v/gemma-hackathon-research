"""[bf16 variant] SRCNN: conv9x9+ReLU -> conv5x5+ReLU -> conv5x5 (NCHW interface, NHWC compute)."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'SRCNN_350',
    'batch': 16,
    'num_channels': 1,
    'height': 1024,
    'width': 1024,
    'c1': 64,
    'c2': 32,
    'k1': 9,
    'k2': 5,
    'k3': 5,
}


def create_inputs(dtype=jnp.bfloat16):
    c = CONFIG
    keys = jax.random.split(jax.random.key(0), 7)
    x = jax.random.uniform(
        keys[0], (c['batch'], c['num_channels'], c['height'], c['width']), dtype)
    w1 = jax.random.normal(keys[1], (c['k1'], c['k1'], c['num_channels'], c['c1']), dtype) * 0.02
    b1 = jax.random.normal(keys[2], (c['c1'],), dtype) * 0.02
    w2 = jax.random.normal(keys[3], (c['k2'], c['k2'], c['c1'], c['c2']), dtype) * 0.02
    b2 = jax.random.normal(keys[4], (c['c2'],), dtype) * 0.02
    w3 = jax.random.normal(keys[5], (c['k3'], c['k3'], c['c2'], c['num_channels']), dtype) * 0.02
    b3 = jax.random.normal(keys[6], (c['num_channels'],), dtype) * 0.02
    return x, w1, b1, w2, b2, w3, b3


def _conv(x, w, b):
    kh, kw = w.shape[0], w.shape[1]
    y = lax.conv_general_dilated(
        x, w,
        window_strides=(1, 1),
        padding=((kh // 2, kh // 2), (kw // 2, kw // 2)),
        dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
    )
    return y + b


def workload(x, w1, b1, w2, b2, w3, b3):
    h = jnp.transpose(x, (0, 2, 3, 1))
    h = jax.nn.relu(_conv(h, w1, b1))
    h = jax.nn.relu(_conv(h, w2, b2))
    h = _conv(h, w3, b3)
    return jnp.transpose(h, (0, 3, 1, 2))
