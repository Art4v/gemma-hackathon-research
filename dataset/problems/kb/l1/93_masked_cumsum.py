"""Masked cumulative sum along a specified dimension."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': '93_masked_cumsum',
    'batch_size': 8192,
    'length': 8192,
    'dim': 1,
}

def create_inputs(dtype=jnp.float32):
    k1, k2 = jax.random.split(jax.random.key(0), 2)
    shape = (CONFIG['batch_size'], CONFIG['length'])
    x = jax.random.uniform(k1, shape, dtype=dtype)
    mask = jax.random.randint(k2, shape, 0, 2).astype(jnp.bool_)
    return (x, mask)

def workload(x, mask):
    return jnp.cumsum(x * mask.astype(x.dtype), axis=CONFIG['dim'])
