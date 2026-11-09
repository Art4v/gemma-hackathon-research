"""Predictive entropy: -sum(softmax(x, dim=1) * log_softmax(x, dim=1), dim=1)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'PredictiveEntropy_7060',
    'batch': 32,
    'channels': 128,
    'height': 128,
    'width': 128,
    'axis': 1,
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    shape = (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width'])
    x = jax.random.uniform(key, shape, dtype)
    return (x,)


def workload(x):
    axis = CONFIG['axis']
    p = jax.nn.softmax(x, axis=axis)
    logp = jax.nn.log_softmax(x, axis=axis)
    return -1.0 * jnp.sum(p * logp, axis=axis)
