"""Conv2d -> HardSwish -> ReLU."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': '69_Conv2d_HardSwish_ReLU',
    'batch_size': 128,
    'in_channels': 8,
    'out_channels': 64,
    'height': 128,
    'width': 128,
    'kernel_size': 3,
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    kx, kw, kb = jax.random.split(key, 3)
    c = CONFIG
    x = jax.random.uniform(kx, (c['batch_size'], c['in_channels'], c['height'], c['width']), dtype)
    w = jax.random.normal(kw, (c['kernel_size'], c['kernel_size'], c['in_channels'], c['out_channels']), dtype) * 0.02
    b = jax.random.normal(kb, (c['out_channels'],), dtype) * 0.02
    return x, w, b


def workload(x, w, b):
    x_nhwc = jnp.transpose(x, (0, 2, 3, 1))
    y = lax.conv_general_dilated(
        x_nhwc, w,
        window_strides=(1, 1),
        padding='VALID',
        dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
    )
    y = y + b
    y = y * jnp.clip(y + 3.0, 0.0, 6.0) / 6.0
    y = jnp.maximum(y, 0.0)
    return jnp.transpose(y, (0, 3, 1, 2))
