"""Tanh: elementwise y = tanh(x)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': '22_Tanh',
    'batch_size': 4096,
    'dim': 16384,
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    x = jax.random.uniform(key, (CONFIG['batch_size'], CONFIG['dim']), dtype=dtype)
    return (x,)


def workload(x):
    return jnp.tanh(x)
