"""Scale: per-channel affine x * weight[c] + bias[c] over NCHW (or NC) input."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'Scale_13874',
    'batch': 8,
    'channels': 128,
    'height': 256,
    'width': 256,
}


def create_inputs(dtype=jnp.float32):
    kx, kw, kb = jax.random.split(jax.random.key(0), 3)
    x = jax.random.uniform(
        kx, (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width']), dtype)
    # ponytail: random (not ones/zeros) so the reference is not an identity map.
    weight = jax.random.normal(kw, (CONFIG['channels'],), dtype) * 0.02
    bias = jax.random.normal(kb, (CONFIG['channels'],), dtype) * 0.02
    return x, weight, bias


def workload(x, weight, bias):
    shape = (1, x.shape[1]) + (1,) * (x.ndim - 2)
    return x * weight.reshape(shape) + bias.reshape(shape)
