"""[m1 M2_second_output variant of kb/l2/87_Conv2d_Subtract_Subtract_Mish] Conv2d -> subtract -> subtract -> Mish, also returning the per-(example, channel) spatial mean of the Mish output (the squeeze step of a squeeze-and-excitation block)."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': '87_Conv2d_Subtract_Subtract_Mish_m1_aux',
    'batch': 64,
    'in_channels': 8,
    'out_channels': 64,
    'height': 256,
    'width': 256,
    'kernel_size': 3,
    'subtract_value_1': 0.5,
    'subtract_value_2': 0.2,
}


def create_inputs(dtype=jnp.float32):
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
    y = y - CONFIG['subtract_value_1']
    y = y - CONFIG['subtract_value_2']
    y = y * jnp.tanh(jax.nn.softplus(y))
    return jnp.transpose(y, (0, 3, 1, 2)), jnp.mean(y, axis=(1, 2))
