"""[bf16 variant] Conv2d -> Subtract -> HardSwish -> MaxPool2d -> Mish."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': '35_Conv2d_Subtract_HardSwish_MaxPool_Mish',
    'batch_size': 64,
    'in_channels': 64,
    'out_channels': 128,
    'height': 128,
    'width': 128,
    'kernel_size': 3,
    'subtract_value': 0.5,
    'pool_kernel_size': 2,
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    kx, kw, kb = jax.random.split(key, 3)
    c = CONFIG
    x = jax.random.uniform(kx, (c['batch_size'], c['in_channels'], c['height'], c['width']), dtype)
    w = jax.random.normal(kw, (c['kernel_size'], c['kernel_size'], c['in_channels'], c['out_channels']), dtype) * 0.02
    b = jax.random.normal(kb, (c['out_channels'],), dtype) * 0.02
    return x, w, b


def workload(x, w, b):
    xh = jnp.transpose(x, (0, 2, 3, 1))
    y = lax.conv_general_dilated(
        xh, w, window_strides=(1, 1), padding='VALID',
        dimension_numbers=('NHWC', 'HWIO', 'NHWC'))
    y = y + b
    y = y - CONFIG['subtract_value']
    y = y * jnp.clip(y + 3.0, 0.0, 6.0) / 6.0
    p = CONFIG['pool_kernel_size']
    y = lax.reduce_window(y, -jnp.inf, lax.max, (1, p, p, 1), (1, p, p, 1), 'VALID')
    y = y * jnp.tanh(jax.nn.softplus(y))
    return jnp.transpose(y, (0, 3, 1, 2))
