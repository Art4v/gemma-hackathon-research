"""GELU activation."""

import jax
import jax.numpy as jnp

CONFIG = {
    'name': '26_GELU_',
    'batch_size': 4096,
    'dim': 16384,
    'approximate': False,
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    x = jax.random.uniform(key, (CONFIG['batch_size'], CONFIG['dim']), dtype=dtype)
    return (x,)


def workload(x):
    return jax.nn.gelu(x, approximate=CONFIG['approximate'])
