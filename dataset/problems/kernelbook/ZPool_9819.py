"""ZPool: max and mean over axis 1, keepdims, concatenated along axis 1."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'ZPool_9819',
    'batch': 8,
    'channels': 128,
    'height': 256,
    'width': 256,
    'pool_axis': 1,
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    shape = (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width'])
    x = jax.random.uniform(key, shape, dtype)
    return (x,)


def workload(x):
    axis = CONFIG['pool_axis']
    max_pool = jnp.max(x, axis=axis, keepdims=True)
    mean_pool = jnp.mean(x, axis=axis, keepdims=True)
    return jnp.concatenate((max_pool, mean_pool), axis=axis)
