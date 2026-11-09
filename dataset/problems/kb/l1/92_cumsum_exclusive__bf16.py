"""[bf16 variant] Exclusive cumulative sum along a specified dimension."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': '92_cumsum_exclusive',
    'batch_size': 8192,
    'seq_len': 8192,
    'dim': 1,
}

def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    x = jax.random.uniform(key, (CONFIG['batch_size'], CONFIG['seq_len']), dtype=dtype)
    return (x,)

def workload(x):
    dim = CONFIG['dim']
    axis = dim if dim >= 0 else x.ndim + dim
    n = x.shape[axis]
    sliced = lax.slice_in_dim(x, 0, n - 1, axis=axis)
    cs = jnp.cumsum(sliced, axis=axis)
    zero_shape = x.shape[:axis] + (1,) + x.shape[axis + 1:]
    zeros = jnp.zeros(zero_shape, dtype=x.dtype)
    return jnp.concatenate([zeros, cs], axis=axis)
