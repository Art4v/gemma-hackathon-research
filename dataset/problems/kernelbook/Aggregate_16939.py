"""Aggregate_16939: sum pooling over one axis (jnp.sum, no mask, no keepdim on output)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'Aggregate_16939',
    'batch': 8,
    'channels': 8,
    'dim_a': 128,
    'dim_b': 128,
    'dim_c': 128,
    'axis': 4,
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    shape = (
        CONFIG['batch'],
        CONFIG['channels'],
        CONFIG['dim_a'],
        CONFIG['dim_b'],
        CONFIG['dim_c'],
    )
    x = jax.random.uniform(key, shape, dtype)
    return (x,)


def workload(x):
    return jnp.sum(x, axis=CONFIG['axis'])
