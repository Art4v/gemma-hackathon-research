"""Scale_2227: per-channel affine, x * weight[c] + bias[c] on NCHW input."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'Scale_2227',
    'batch': 8,
    'channels': 128,
    'height': 256,
    'width': 256,
}


def create_inputs(dtype=jnp.float32):
    kx, kw, kb = jax.random.split(jax.random.key(0), 3)
    b, c, h, w = CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width']
    x = jax.random.uniform(kx, (b, c, h, w), dtype)
    weight = jax.random.normal(kw, (c,), dtype) * 0.02
    bias = jax.random.normal(kb, (c,), dtype) * 0.02
    return x, weight, bias


def workload(x, weight, bias):
    c = x.shape[1]
    return x * weight.reshape(1, c, 1, 1) + bias.reshape(1, c, 1, 1)
