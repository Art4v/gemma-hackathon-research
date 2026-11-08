"""Elementwise Softplus activation."""

import jax
import jax.numpy as jnp

CONFIG = {
    'name': '29_Softplus',
    'batch_size': 4096,
    'dim': 16384,
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    shape = (CONFIG['batch_size'], CONFIG['dim'])
    x = jax.random.uniform(key, shape=shape, dtype=dtype)
    return (x,)


def workload(x):
    return jax.nn.softplus(x)
