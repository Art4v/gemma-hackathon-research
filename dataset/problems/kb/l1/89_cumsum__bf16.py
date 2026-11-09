"""[bf16 variant] Cumulative sum (prefix sum) along a specified dimension."""

import jax
import jax.numpy as jnp

CONFIG = {
    'name': '89_cumsum',
    'batch_size': 4096,
    'length': 32768,
    'dim': 1,
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    shape = (CONFIG['batch_size'], CONFIG['length'])
    x = jax.random.uniform(key, shape, dtype=dtype)
    return (x,)


def workload(x):
    return jnp.cumsum(x, axis=CONFIG['dim'])
