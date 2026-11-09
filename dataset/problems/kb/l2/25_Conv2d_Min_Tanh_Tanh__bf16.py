"""[bf16 variant] Conv2d (valid, 3x3) -> min over channels (keepdim) -> tanh -> tanh."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': '25_Conv2d_Min_Tanh_Tanh',
    'batch': 32,
    'in_channels': 16,
    'out_channels': 64,
    'height': 256,
    'width': 256,
    'kernel_size': 3,
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    kx, kw, kb = jax.random.split(key, 3)
    c = CONFIG
    x = jax.random.uniform(kx, (c['batch'], c['in_channels'], c['height'], c['width']), dtype)
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
    y = jnp.min(y, axis=-1, keepdims=True)
    y = jnp.tanh(y)
    y = jnp.tanh(y)
    return jnp.transpose(y, (0, 3, 1, 2))
