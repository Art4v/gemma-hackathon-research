"""[bf16 variant] SpatialAttentionGate: conv1x1 -> ReLU -> conv1x1 -> sigmoid."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'SpatialAttentionGate_5270',
    'batch': 16,
    'channel': 128,
    'reduction': 16,
    'height': 128,
    'width': 128,
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    kx, kw1, kb1, kw2, kb2 = jax.random.split(key, 5)
    c, r = CONFIG['channel'], CONFIG['reduction']
    x = jax.random.uniform(
        kx, (CONFIG['batch'], c, CONFIG['height'], CONFIG['width']), dtype)
    w1 = jax.random.normal(kw1, (1, 1, c, r), dtype) * 0.02
    b1 = jax.random.normal(kb1, (r,), dtype) * 0.02
    w2 = jax.random.normal(kw2, (1, 1, r, 1), dtype) * 0.02
    b2 = jax.random.normal(kb2, (1,), dtype) * 0.02
    return x, w1, b1, w2, b2


def _conv1x1(x, w, b):
    y = lax.conv_general_dilated(
        x, w, window_strides=(1, 1), padding='VALID',
        dimension_numbers=('NHWC', 'HWIO', 'NHWC'))
    return y + b


def workload(x, w1, b1, w2, b2):
    h = jnp.transpose(x, (0, 2, 3, 1))
    h = jax.nn.relu(_conv1x1(h, w1, b1))
    h = jax.nn.sigmoid(_conv1x1(h, w2, b2))
    return jnp.transpose(h, (0, 3, 1, 2))
