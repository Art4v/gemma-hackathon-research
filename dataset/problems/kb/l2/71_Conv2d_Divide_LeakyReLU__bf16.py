"""[bf16 variant] Conv2d (VALID, stride 1, bias) -> divide by constant -> LeakyReLU(0.01)."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': '71_Conv2d_Divide_LeakyReLU',
    'batch_size': 128,
    'in_channels': 8,
    'out_channels': 64,
    'height': 128,
    'width': 128,
    'kernel_size': 3,
    'divisor': 2.0,
    'negative_slope': 0.01,
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    kx, kw, kb = jax.random.split(key, 3)
    c = CONFIG
    x = jax.random.uniform(
        kx, (c['batch_size'], c['in_channels'], c['height'], c['width']), dtype)
    w = jax.random.normal(
        kw, (c['kernel_size'], c['kernel_size'], c['in_channels'], c['out_channels']),
        dtype) * 0.02
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
    y = y + b.reshape(1, 1, 1, -1)
    y = y / CONFIG['divisor']
    y = jnp.where(y >= 0, y, y * CONFIG['negative_slope'])
    return jnp.transpose(y, (0, 3, 1, 2))
