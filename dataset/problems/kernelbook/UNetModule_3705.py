"""UNetModule: Conv3x3(pad=1) -> SELU -> Conv3x3(pad=1) -> SELU."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'UNetModule_3705',
    'batch': 8,
    'in_channels': 128,
    'out_channels': 128,
    'height': 256,
    'width': 256,
    'kernel_size': 3,
}


def create_inputs(dtype=jnp.float32):
    c = CONFIG
    k = c['kernel_size']
    kx, kw1, kb1, kw2, kb2 = jax.random.split(jax.random.key(0), 5)
    x = jax.random.uniform(kx, (c['batch'], c['in_channels'], c['height'], c['width']), dtype)
    w1 = jax.random.normal(kw1, (k, k, c['in_channels'], c['out_channels']), dtype) * 0.02
    b1 = jax.random.normal(kb1, (c['out_channels'],), dtype) * 0.02
    w2 = jax.random.normal(kw2, (k, k, c['out_channels'], c['out_channels']), dtype) * 0.02
    b2 = jax.random.normal(kb2, (c['out_channels'],), dtype) * 0.02
    return x, w1, b1, w2, b2


def _conv_selu(x, w, b):
    # SAME == padding=1 for a 3x3 kernel at stride 1.
    y = lax.conv_general_dilated(
        x, w, window_strides=(1, 1), padding='SAME',
        dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
        precision=lax.Precision.HIGHEST)
    return jax.nn.selu(y + b)


def workload(x, w1, b1, w2, b2):
    h = jnp.transpose(x, (0, 2, 3, 1))
    h = _conv_selu(h, w1, b1)
    h = _conv_selu(h, w2, b2)
    return jnp.transpose(h, (0, 3, 1, 2))
