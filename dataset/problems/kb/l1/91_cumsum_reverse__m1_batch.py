"""[m1 M1_batch variant of kb/l1/91_cumsum_reverse] Reverse cumulative sum along the row axis, applied over a new leading axis of size 4 with the rows shrunk from 8192 to 2048."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': '91_cumsum_reverse_m1_batch',
    'outer_batch': 4,
    'batch_size': 2048,
    'length': 8192,
    'dim': 1,
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    x = jax.random.uniform(
        key, (CONFIG['outer_batch'], CONFIG['batch_size'], CONFIG['length']), dtype=dtype)
    return (x,)


def _reverse_cumsum(x):
    dim = CONFIG['dim']
    return jnp.flip(jnp.cumsum(jnp.flip(x, axis=dim), axis=dim), axis=dim)


def workload(x):
    return jax.vmap(_reverse_cumsum)(x)
