"""PreNet: Conv2d(3x3, pad 1) -> LeakyReLU(0.2) -> Conv2d(3x3, pad 1) -> LeakyReLU(0.2)."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'PreNet_7012',
    'batch': 32,
    'in_channels': 4,
    'mid_channels': 8,
    'out_channels': 16,
    'height': 512,
    'width': 512,
    'kernel_size': 3,
    'padding': 1,
    'negative_slope': 0.2,
}


def create_inputs(dtype=jnp.float32):
    c = CONFIG
    k = c['kernel_size']
    k_x, k_w1, k_b1, k_w2, k_b2 = jax.random.split(jax.random.key(0), 5)
    labels = jax.random.uniform(
        k_x, (c['batch'], c['in_channels'], c['height'], c['width']), dtype)
    w1 = jax.random.normal(k_w1, (k, k, c['in_channels'], c['mid_channels']), dtype) * 0.02
    b1 = jax.random.normal(k_b1, (c['mid_channels'],), dtype) * 0.02
    w2 = jax.random.normal(k_w2, (k, k, c['mid_channels'], c['out_channels']), dtype) * 0.02
    b2 = jax.random.normal(k_b2, (c['out_channels'],), dtype) * 0.02
    return labels, w1, b1, w2, b2


def _conv_leaky(x, w, b):
    p = CONFIG['padding']
    y = lax.conv_general_dilated(
        x, w,
        window_strides=(1, 1),
        padding=((p, p), (p, p)),
        dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
        precision=lax.Precision.HIGHEST,
    ) + b
    return jax.nn.leaky_relu(y, CONFIG['negative_slope'])


def workload(labels, w1, b1, w2, b2):
    x = jnp.transpose(labels, (0, 2, 3, 1))
    x = _conv_leaky(x, w1, b1)
    x = _conv_leaky(x, w2, b2)
    return jnp.transpose(x, (0, 3, 1, 2))
