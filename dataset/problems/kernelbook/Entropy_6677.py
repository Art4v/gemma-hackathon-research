"""Entropy: x * log(x) with NaNs zeroed, negated sum over the last axis."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'Entropy_6677',
    'batch': 8,
    'height': 128,
    'width': 128,
    'channels': 256,
    'axis': -1,
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    shape = (CONFIG['batch'], CONFIG['height'], CONFIG['width'], CONFIG['channels'])
    x = jax.random.uniform(key, shape, dtype)
    return (x,)


def workload(x):
    plogp = x * jnp.log(x)
    plogp = jnp.where(plogp != plogp, jnp.zeros_like(plogp), plogp)
    return -jnp.sum(plogp, axis=CONFIG['axis'])
