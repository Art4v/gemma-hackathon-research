"""ToRGB: Conv2d(C->3, kernel 1x1, stride 1, padding 0, bias) -> tanh."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'ToRGB_4649',
    'batch': 16,
    'in_channels': 256,
    'out_channels': 3,
    'height': 128,
    'width': 128,
    'kernel_size': 1,
    'stride': 1,
    'padding': 0,
}


def create_inputs(dtype=jnp.float32):
    k_x, k_w, k_b = jax.random.split(jax.random.key(0), 3)
    c = CONFIG
    x = jax.random.uniform(
        k_x, (c['batch'], c['in_channels'], c['height'], c['width']), dtype)
    w = jax.random.normal(
        k_w, (c['kernel_size'], c['kernel_size'], c['in_channels'], c['out_channels']),
        dtype) * 0.02
    b = jax.random.normal(k_b, (c['out_channels'],), dtype) * 0.02
    return x, w, b


def workload(x, w, b):
    """x: (N, C, H, W); w: HWIO (k, k, C, 3); b: (3,). Returns (N, 3, H', W')."""
    p = CONFIG['padding']
    s = CONFIG['stride']
    x_nhwc = jnp.transpose(x, (0, 2, 3, 1))
    y = lax.conv_general_dilated(
        x_nhwc, w,
        window_strides=(s, s),
        padding=((p, p), (p, p)),
        dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
    )
    y = jnp.tanh(y + b)
    return jnp.transpose(y, (0, 3, 1, 2))
