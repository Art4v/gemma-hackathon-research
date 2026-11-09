"""MonomialNN: concat([x**d for d in degrees], axis=1)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'MonomialNN_5069',
    'batch': 16,
    'channels': 128,
    'height': 128,
    'width': 128,
    'degrees': (1, 2, 3, 4),
}


def create_inputs(dtype=jnp.float32):
    shape = (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width'])
    x = jax.random.uniform(jax.random.key(0), shape, dtype)
    return (x,)


def workload(x):
    return jnp.concatenate([x ** d for d in CONFIG['degrees']], axis=1)
