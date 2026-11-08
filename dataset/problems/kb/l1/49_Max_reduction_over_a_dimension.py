"""Max reduction over a specific dimension."""

import jax
import jax.numpy as jnp

CONFIG = {
    'name': '49_Max_reduction_over_a_dimension',
    'batch_size': 64,
    'dim1': 1024,
    'dim2': 1024,
    'dim': 1,
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    shape = (CONFIG['batch_size'], CONFIG['dim1'], CONFIG['dim2'])
    x = jax.random.uniform(key, shape=shape, dtype=dtype)
    return (x,)


def workload(x):
    return jnp.max(x, axis=CONFIG['dim'])
