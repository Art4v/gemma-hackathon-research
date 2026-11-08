"""Cumulative product along a specified axis."""

import jax
import jax.numpy as jnp

CONFIG = {
    'name': '90_cumprod',
    'batch_size': 8192,
    'length': 8192,
    'dim': 1,
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    shape = (CONFIG['batch_size'], CONFIG['length'])
    x = jax.random.uniform(key, shape, dtype=dtype)
    return (x,)


def workload(x):
    return jnp.cumprod(x, axis=CONFIG['dim'])
