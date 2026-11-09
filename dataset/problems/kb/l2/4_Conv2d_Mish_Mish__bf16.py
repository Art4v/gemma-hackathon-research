"""[bf16 variant] Conv2d (valid, with bias) -> Mish -> Mish."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': '4_Conv2d_Mish_Mish',
    'batch': 16,
    'in_channels': 64,
    'out_channels': 128,
    'height': 256,
    'width': 256,
    'kernel_size': 3,
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    kx, kw, kb = jax.random.split(key, 3)
    c = CONFIG
    x = jax.random.uniform(kx, (c['batch'], c['in_channels'], c['height'], c['width']), dtype)
    k = c['kernel_size']
    w = jax.random.normal(kw, (k, k, c['in_channels'], c['out_channels']), dtype) * 0.02
    b = jax.random.normal(kb, (c['out_channels'],), dtype) * 0.02
    return x, w, b


def _mish(x):
    return x * jnp.tanh(jax.nn.softplus(x))


def workload(x, w, b):
    x_nhwc = jnp.transpose(x, (0, 2, 3, 1))
    y = lax.conv_general_dilated(
        x_nhwc, w,
        window_strides=(1, 1),
        padding='VALID',
        dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
    )
    y = y + b
    y = _mish(y)
    y = _mish(y)
    return jnp.transpose(y, (0, 3, 1, 2))
