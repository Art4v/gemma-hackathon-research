"""Sigmoid activation: y = sigmoid(x)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': '21_Sigmoid',
    'batch': 4096,
    'dim': 16384,
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    x = jax.random.uniform(key, (CONFIG['batch'], CONFIG['dim']), dtype=dtype)
    return (x,)


def workload(x):
    return jax.nn.sigmoid(x)
