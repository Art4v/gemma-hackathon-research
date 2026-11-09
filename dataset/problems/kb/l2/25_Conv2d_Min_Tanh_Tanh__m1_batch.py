"""[m1 M1_batch variant of kb/l2/25_Conv2d_Min_Tanh_Tanh] Conv2d (valid, 3x3) -> min over channels -> tanh -> tanh, vmapped over a new leading group axis of size 4 with shared conv weights and 128x128 images."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': '25_Conv2d_Min_Tanh_Tanh_m1_batch',
    'groups': 4,
    'batch': 32,
    'in_channels': 16,
    'out_channels': 64,
    'height': 128,
    'width': 128,
    'kernel_size': 3,
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    kx, kw, kb = jax.random.split(key, 3)
    c = CONFIG
    x = jax.random.uniform(kx, (c['groups'], c['batch'], c['in_channels'], c['height'], c['width']), dtype)
    w = jax.random.normal(kw, (c['kernel_size'], c['kernel_size'], c['in_channels'], c['out_channels']), dtype) * 0.02
    b = jax.random.normal(kb, (c['out_channels'],), dtype) * 0.02
    return x, w, b


def _single(x, w, b):
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


def workload(x, w, b):
    return jax.vmap(_single, in_axes=(0, None, None))(x, w, b)
