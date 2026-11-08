"""Exclusive cumulative sum: cumsum(x[..., :-1]) prepended with zeros along dim."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': '92_cumsum_exclusive',
    'batch_size': 4096,
    'length': 16384,
    'dim': 1,
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    x = jax.random.uniform(key, (CONFIG['batch_size'], CONFIG['length']), dtype=dtype)
    return (x,)


def workload(x):
    dim = CONFIG['dim']
    n = x.shape[dim]
    head = lax.slice_in_dim(x, 0, n - 1, axis=dim)
    cs = jnp.cumsum(head, axis=dim)
    zeros = jnp.zeros_like(lax.slice_in_dim(x, 0, 1, axis=dim))
    return jnp.concatenate([zeros, cs], axis=dim)
