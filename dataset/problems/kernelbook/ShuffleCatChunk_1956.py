"""ShuffleCatChunk: chunk two NCHW tensors by channel, interleave a/b channels, concatenate along dim 1."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'ShuffleCatChunk_1956',
    'batch': 8,
    'channels': 128,
    'height': 256,
    'width': 256,
}


def create_inputs(dtype=jnp.float32):
    shape = (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width'])
    key_a, key_b = jax.random.split(jax.random.key(0))
    a = jax.random.uniform(key_a, shape, dtype)
    b = jax.random.uniform(key_b, shape, dtype)
    return a, b


def workload(a, b):
    n, c, h, w = a.shape
    return jnp.stack((a, b), axis=2).reshape(n, 2 * c, h, w)
