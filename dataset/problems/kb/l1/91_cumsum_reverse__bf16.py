"""[bf16 variant] Reverse cumulative sum along a specified dimension."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': '91_cumsum_reverse',
    'batch_size': 8192,
    'length': 8192,
    'dim': 1,
}

def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    x = jax.random.uniform(key, (CONFIG['batch_size'], CONFIG['length']), dtype=dtype)
    return (x,)

def workload(x):
    dim = CONFIG['dim']
    return jnp.flip(jnp.cumsum(jnp.flip(x, axis=dim), axis=dim), axis=dim)
