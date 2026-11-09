"""SAM: 1x1 Conv2d (C->1) -> sigmoid -> broadcast multiply with input."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'SAM_15078',
    'batch': 16,
    'in_channels': 128,
    'height': 128,
    'width': 128,
    'out_channels': 1,
    'kernel_size': 1,
}


def create_inputs(dtype=jnp.float32):
    k_x, k_w, k_b = jax.random.split(jax.random.key(0), 3)
    x = jax.random.uniform(
        k_x, (CONFIG['batch'], CONFIG['in_channels'], CONFIG['height'], CONFIG['width']), dtype)
    k = CONFIG['kernel_size']
    w = jax.random.normal(k_w, (k, k, CONFIG['in_channels'], CONFIG['out_channels']), dtype) * 0.02
    b = jax.random.normal(k_b, (CONFIG['out_channels'],), dtype) * 0.02
    return x, w, b


def workload(x, w, b):
    x_nhwc = jnp.transpose(x, (0, 2, 3, 1))
    y = lax.conv_general_dilated(
        x_nhwc, w, window_strides=(1, 1), padding='VALID',
        dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
        precision=lax.Precision.HIGHEST,
    ) + b
    attention = jax.nn.sigmoid(jnp.transpose(y, (0, 3, 1, 2)))
    return attention * x
