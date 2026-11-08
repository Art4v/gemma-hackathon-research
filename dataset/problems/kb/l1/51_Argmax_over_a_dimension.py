"""Argmax over a dimension (dim=1) of a 3D tensor."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': '51_Argmax_over_a_dimension',
    'batch_size': 16,
    'dim1': 2048,
    'dim2': 2048,
    'dim': 1,
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    x = jax.random.uniform(
        key, (CONFIG['batch_size'], CONFIG['dim1'], CONFIG['dim2']), dtype=dtype
    )
    return (x,)


def workload(x):
    return jnp.argmax(x, axis=CONFIG['dim'])
