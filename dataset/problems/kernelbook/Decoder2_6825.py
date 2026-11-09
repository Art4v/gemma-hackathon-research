"""Decoder2: reflect-pad -> conv3x3(128->64) -> ReLU -> nearest upsample x2 -> reflect-pad -> conv3x3(64->64) -> ReLU -> reflect-pad -> conv3x3(64->3)."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'Decoder2_6825',
    'batch': 8,
    'in_channels': 128,
    'mid_channels': 64,
    'out_channels': 3,
    'height': 128,
    'width': 128,
    'kernel_size': 3,
    'pad': 1,
    'scale_factor': 2,
}


def create_inputs(dtype=jnp.float32):
    c = CONFIG
    k = c['kernel_size']
    keys = jax.random.split(jax.random.key(0), 7)
    x = jax.random.uniform(keys[0], (c['batch'], c['in_channels'], c['height'], c['width']), dtype)
    w5 = jax.random.normal(keys[1], (k, k, c['in_channels'], c['mid_channels']), dtype) * 0.02
    b5 = jax.random.normal(keys[2], (c['mid_channels'],), dtype) * 0.02
    w6 = jax.random.normal(keys[3], (k, k, c['mid_channels'], c['mid_channels']), dtype) * 0.02
    b6 = jax.random.normal(keys[4], (c['mid_channels'],), dtype) * 0.02
    w7 = jax.random.normal(keys[5], (k, k, c['mid_channels'], c['out_channels']), dtype) * 0.02
    b7 = jax.random.normal(keys[6], (c['out_channels'],), dtype) * 0.02
    return x, w5, b5, w6, b6, w7, b7


def _reflect_pad(x):
    p = CONFIG['pad']
    return jnp.pad(x, ((0, 0), (p, p), (p, p), (0, 0)), mode='reflect')


def _conv(x, w, b):
    y = lax.conv_general_dilated(
        x, w, window_strides=(1, 1), padding='VALID',
        dimension_numbers=('NHWC', 'HWIO', 'NHWC'))
    return y + b


def workload(x, w5, b5, w6, b6, w7, b7):
    s = CONFIG['scale_factor']
    out = jnp.transpose(x, (0, 2, 3, 1))
    out = jax.nn.relu(_conv(_reflect_pad(out), w5, b5))
    out = jnp.repeat(jnp.repeat(out, s, axis=1), s, axis=2)
    out = jax.nn.relu(_conv(_reflect_pad(out), w6, b6))
    out = _conv(_reflect_pad(out), w7, b7)
    return jnp.transpose(out, (0, 3, 1, 2))
