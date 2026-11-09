"""QueryModule: feats * attn (broadcast over channels) -> Conv2d 3x3 -> ReLU -> Conv2d 3x3 -> ReLU."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'QueryModule_6298',
    'batch': 8,
    'dim': 128,
    'height': 128,
    'width': 128,
    'kernel_size': 3,
}


def create_inputs(dtype=jnp.float32):
    b, c = CONFIG['batch'], CONFIG['dim']
    h, w, k = CONFIG['height'], CONFIG['width'], CONFIG['kernel_size']
    keys = jax.random.split(jax.random.key(0), 6)
    feats = jax.random.uniform(keys[0], (b, c, h, w), dtype)
    attn = jax.random.uniform(keys[1], (b, 1, h, w), dtype)
    w1 = jax.random.normal(keys[2], (k, k, c, c), dtype) * 0.02
    b1 = jax.random.normal(keys[3], (c,), dtype) * 0.02
    w2 = jax.random.normal(keys[4], (k, k, c, c), dtype) * 0.02
    b2 = jax.random.normal(keys[5], (c,), dtype) * 0.02
    return feats, attn, w1, b1, w2, b2


def _conv_relu(x, w, b):
    y = lax.conv_general_dilated(
        x, w, window_strides=(1, 1), padding='VALID',
        dimension_numbers=('NHWC', 'HWIO', 'NHWC'))
    return jax.nn.relu(y + b)


def workload(feats, attn, w1, b1, w2, b2):
    x = feats * attn  # broadcast over channel == attn.repeat(1, dim, 1, 1)
    x = jnp.transpose(x, (0, 2, 3, 1))
    x = _conv_relu(x, w1, b1)
    x = _conv_relu(x, w2, b2)
    return jnp.transpose(x, (0, 3, 1, 2))
