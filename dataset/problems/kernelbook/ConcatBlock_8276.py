"""ConcatBlock: Conv2d 1x1 -> LeakyReLU -> Conv2d 1x1 -> LeakyReLU."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'ConcatBlock_8276',
    'batch': 8,
    'in_channels': 128,
    'out_channels': 128,
    'height': 256,
    'width': 256,
    'kernel_size': 1,
    'negative_slope': 0.01,
}


def create_inputs(dtype=jnp.float32):
    k = jax.random.split(jax.random.key(0), 5)
    c_in, c_out, ks = CONFIG['in_channels'], CONFIG['out_channels'], CONFIG['kernel_size']
    x = jax.random.uniform(
        k[0], (CONFIG['batch'], c_in, CONFIG['height'], CONFIG['width']), dtype)
    w1 = jax.random.normal(k[1], (ks, ks, c_in, c_in), dtype) * 0.02
    b1 = jax.random.normal(k[2], (c_in,), dtype) * 0.02
    w2 = jax.random.normal(k[3], (ks, ks, c_in, c_out), dtype) * 0.02
    b2 = jax.random.normal(k[4], (c_out,), dtype) * 0.02
    return x, w1, b1, w2, b2


def _conv_lrelu(h, w, b):
    y = lax.conv_general_dilated(
        h, w, window_strides=(1, 1), padding='VALID',
        dimension_numbers=('NHWC', 'HWIO', 'NHWC'))
    y = y + b
    return jnp.where(y >= 0, y, CONFIG['negative_slope'] * y)


def workload(x, w1, b1, w2, b2):
    h = jnp.transpose(x, (0, 2, 3, 1))
    h = _conv_lrelu(h, w1, b1)
    h = _conv_lrelu(h, w2, b2)
    return jnp.transpose(h, (0, 3, 1, 2))
