"""[bf16 variant] Conv2d(k9) -> ReLU -> Conv2d(k5) -> ReLU -> Conv2d(k5), SRCNN-style NetModel."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'NetModel_6491',
    'batch': 16,
    'in_channels': 1,
    'height': 1024,
    'width': 1024,
    'n1': 4,
    'n2': 4,
    'out_channels': 1,
    'k1': 9,
    'k2': 5,
    'k3': 5,
}


def create_inputs(dtype=jnp.bfloat16):
    c = CONFIG
    k = jax.random.split(jax.random.key(0), 7)
    x = jax.random.uniform(k[0], (c['batch'], c['in_channels'], c['height'], c['width']), dtype)
    w1 = jax.random.normal(k[1], (c['k1'], c['k1'], c['in_channels'], c['n1']), dtype) * 0.02
    b1 = jax.random.normal(k[2], (c['n1'],), dtype) * 0.02
    w2 = jax.random.normal(k[3], (c['k2'], c['k2'], c['n1'], c['n2']), dtype) * 0.02
    b2 = jax.random.normal(k[4], (c['n2'],), dtype) * 0.02
    w3 = jax.random.normal(k[5], (c['k3'], c['k3'], c['n2'], c['out_channels']), dtype) * 0.02
    b3 = jax.random.normal(k[6], (c['out_channels'],), dtype) * 0.02
    return x, w1, b1, w2, b2, w3, b3


def _conv_same(x, w, b):
    # Stride 1, padding k//2 on each side, matching the PyTorch module.
    ph, pw = w.shape[0] // 2, w.shape[1] // 2
    y = lax.conv_general_dilated(
        x, w, window_strides=(1, 1), padding=((ph, ph), (pw, pw)),
        dimension_numbers=('NHWC', 'HWIO', 'NHWC'))
    return y + b


def workload(x, w1, b1, w2, b2, w3, b3):
    h = jnp.transpose(x, (0, 2, 3, 1))
    h = jax.nn.relu(_conv_same(h, w1, b1))
    h = jax.nn.relu(_conv_same(h, w2, b2))
    h = _conv_same(h, w3, b3)
    return jnp.transpose(h, (0, 3, 1, 2))
