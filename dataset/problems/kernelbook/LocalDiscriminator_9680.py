"""[bf16 variant] LocalDiscriminator: 1x1 conv -> ReLU -> 1x1 conv -> ReLU -> 1x1 conv (NCHW in/out)."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'LocalDiscriminator_9680',
    'batch': 8,
    'y_size': 64,
    'M_channels': 64,
    'hidden': 256,
    'out_channels': 1,
    'height': 128,
    'width': 128,
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    kx, k0, k1, k2 = jax.random.split(key, 4)
    c_in = CONFIG['y_size'] + CONFIG['M_channels']
    hid = CONFIG['hidden']
    c_out = CONFIG['out_channels']
    x = jax.random.uniform(
        kx, (CONFIG['batch'], c_in, CONFIG['height'], CONFIG['width']), dtype)
    w0 = jax.random.normal(k0, (1, 1, c_in, hid), dtype) * 0.02
    b0 = jnp.zeros((hid,), dtype)
    w1 = jax.random.normal(k1, (1, 1, hid, hid), dtype) * 0.02
    b1 = jnp.zeros((hid,), dtype)
    w2 = jax.random.normal(k2, (1, 1, hid, c_out), dtype) * 0.02
    b2 = jnp.zeros((c_out,), dtype)
    return x, w0, b0, w1, b1, w2, b2


def _conv1x1(h, w, b):
    y = lax.conv_general_dilated(
        h, w, window_strides=(1, 1), padding='VALID',
        dimension_numbers=('NHWC', 'HWIO', 'NHWC'))
    return y + b


def workload(x, w0, b0, w1, b1, w2, b2):
    h = jnp.transpose(x, (0, 2, 3, 1))
    h = jax.nn.relu(_conv1x1(h, w0, b0))
    h = jax.nn.relu(_conv1x1(h, w1, b1))
    h = _conv1x1(h, w2, b2)
    return jnp.transpose(h, (0, 3, 1, 2))
