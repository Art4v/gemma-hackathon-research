"""[m1 M1_batch variant of kb/l1/92_cumsum_exclusive] Exclusive cumulative sum along the row axis, applied over a new leading axis of size 4 with the rows shrunk from 4096 to 1024."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': '92_cumsum_exclusive_m1_batch',
    'outer_batch': 4,
    'batch_size': 1024,
    'length': 16384,
    'dim': 1,
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    x = jax.random.uniform(
        key, (CONFIG['outer_batch'], CONFIG['batch_size'], CONFIG['length']), dtype=dtype)
    return (x,)


def _exclusive_cumsum(x):
    dim = CONFIG['dim']
    n = x.shape[dim]
    head = lax.slice_in_dim(x, 0, n - 1, axis=dim)
    cs = jnp.cumsum(head, axis=dim)
    zeros = jnp.zeros_like(lax.slice_in_dim(x, 0, 1, axis=dim))
    return jnp.concatenate([zeros, cs], axis=dim)


def workload(x):
    return jax.vmap(_exclusive_cumsum)(x)
