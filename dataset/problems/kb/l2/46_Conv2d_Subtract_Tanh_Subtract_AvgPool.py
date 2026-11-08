"""Conv2d -> subtract -> tanh -> subtract -> AvgPool2d."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': '46_Conv2d_Subtract_Tanh_Subtract_AvgPool',
    'batch': 128,
    'in_channels': 64,
    'out_channels': 128,
    'height': 128,
    'width': 128,
    'kernel_size': 3,
    'subtract1_value': 0.5,
    'subtract2_value': 0.2,
    'kernel_size_pool': 2,
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    kx, kw, kb = jax.random.split(key, 3)
    c = CONFIG
    x = jax.random.uniform(kx, (c['batch'], c['in_channels'], c['height'], c['width']), dtype)
    k = c['kernel_size']
    w = jax.random.normal(kw, (k, k, c['in_channels'], c['out_channels']), dtype) * 0.02
    b = jax.random.normal(kb, (c['out_channels'],), dtype) * 0.02
    return x, w, b


def workload(x, w, b):
    c = CONFIG
    x = jnp.transpose(x, (0, 2, 3, 1))  # NCHW -> NHWC
    y = lax.conv_general_dilated(
        x, w,
        window_strides=(1, 1),
        padding='VALID',
        dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
    )
    y = y + b
    y = y - c['subtract1_value']
    y = jnp.tanh(y)
    y = y - c['subtract2_value']
    p = c['kernel_size_pool']
    y = lax.reduce_window(
        y, jnp.array(0, y.dtype), lax.add,
        window_dimensions=(1, p, p, 1),
        window_strides=(1, p, p, 1),
        padding='VALID',
    ) / (p * p)
    return jnp.transpose(y, (0, 3, 1, 2))  # NHWC -> NCHW
