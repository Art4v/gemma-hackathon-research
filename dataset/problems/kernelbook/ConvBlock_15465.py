"""Conv2d (k=4, stride=1, pad=4, bias) -> PReLU (single shared slope)."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'ConvBlock_15465',
    'batch': 16,
    'in_channels': 128,
    'out_channels': 128,
    'height': 128,
    'width': 128,
    'kernel_size': 4,
    'stride': 1,
    'padding': 4,
    'prelu_init': 0.25,
}


def create_inputs(dtype=jnp.float32):
    c = CONFIG
    k_x, k_w, k_b = jax.random.split(jax.random.key(0), 3)
    x = jax.random.uniform(
        k_x, (c['batch'], c['in_channels'], c['height'], c['width']), dtype)
    w = jax.random.normal(
        k_w, (c['out_channels'], c['in_channels'], c['kernel_size'], c['kernel_size']),
        dtype) * 0.02
    b = jax.random.normal(k_b, (c['out_channels'],), dtype) * 0.02
    alpha = jnp.full((1,), c['prelu_init'], dtype)
    return x, w, b, alpha


def workload(x, w, b, alpha):
    s, p = CONFIG['stride'], CONFIG['padding']
    x_nhwc = jnp.transpose(x, (0, 2, 3, 1))
    y = lax.conv_general_dilated(
        x_nhwc, w,
        window_strides=(s, s),
        padding=((p, p), (p, p)),
        dimension_numbers=('NHWC', 'OIHW', 'NHWC'),
    )
    y = y + b
    y = jnp.where(y >= 0, y, alpha[0] * y)
    return jnp.transpose(y, (0, 3, 1, 2))
