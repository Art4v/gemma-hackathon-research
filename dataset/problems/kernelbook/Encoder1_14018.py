"""[bf16 variant] Encoder1: conv1x1 -> reflection pad 1 -> conv3x3 (valid) -> ReLU."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'Encoder1_14018',
    'batch': 8,
    'in_channels': 3,
    'mid_channels': 3,
    'out_channels': 64,
    'height': 1024,
    'width': 1024,
    'kernel0': 1,
    'kernel1': 3,
    'pad': 1,
}


def create_inputs(dtype=jnp.bfloat16):
    c = CONFIG
    k = jax.random.split(jax.random.key(0), 5)
    x = jax.random.uniform(k[0], (c['batch'], c['in_channels'], c['height'], c['width']), dtype)
    w0 = jax.random.normal(k[1], (c['kernel0'], c['kernel0'], c['in_channels'], c['mid_channels']), dtype) * 0.02
    b0 = jax.random.normal(k[2], (c['mid_channels'],), dtype) * 0.02
    w1 = jax.random.normal(k[3], (c['kernel1'], c['kernel1'], c['mid_channels'], c['out_channels']), dtype) * 0.02
    b1 = jax.random.normal(k[4], (c['out_channels'],), dtype) * 0.02
    return x, w0, b0, w1, b1


def _conv(x, w, b):
    y = lax.conv_general_dilated(
        x, w, window_strides=(1, 1), padding='VALID',
        dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
        precision=lax.Precision.HIGHEST)
    return y + b


def workload(x, w0, b0, w1, b1):
    p = CONFIG['pad']
    h = jnp.transpose(x, (0, 2, 3, 1))
    h = _conv(h, w0, b0)
    h = jnp.pad(h, ((0, 0), (p, p), (p, p), (0, 0)), mode='reflect')
    h = jax.nn.relu(_conv(h, w1, b1))
    return jnp.transpose(h, (0, 3, 1, 2))
