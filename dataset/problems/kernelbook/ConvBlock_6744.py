"""[bf16 variant] Conv2d (k=4, s=1, p=1, bias) -> LeakyReLU(0.01)."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'ConvBlock_6744',
    'batch': 8,
    'in_channels': 128,
    'out_channels': 128,
    'height': 128,
    'width': 128,
    'kernel_size': 4,
    'stride': 1,
    'padding': 1,
    'negative_slope': 0.01,
}


def create_inputs(dtype=jnp.bfloat16):
    c = CONFIG
    kx, kw, kb = jax.random.split(jax.random.key(0), 3)
    x = jax.random.uniform(kx, (c['batch'], c['in_channels'], c['height'], c['width']), dtype)
    w = jax.random.normal(kw, (c['out_channels'], c['in_channels'], c['kernel_size'], c['kernel_size']), dtype) * 0.02
    b = jax.random.normal(kb, (c['out_channels'],), dtype) * 0.02
    return x, w, b


def workload(x, w, b):
    p = CONFIG['padding']
    s = CONFIG['stride']
    x_nhwc = jnp.transpose(x, (0, 2, 3, 1))
    w_hwio = jnp.transpose(w, (2, 3, 1, 0))
    y = lax.conv_general_dilated(
        x_nhwc, w_hwio,
        window_strides=(s, s),
        padding=((p, p), (p, p)),
        dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
        precision=lax.Precision.HIGHEST,
    )
    y = y + b
    y = jnp.where(y >= 0, y, y * CONFIG['negative_slope'])
    return jnp.transpose(y, (0, 3, 1, 2))
