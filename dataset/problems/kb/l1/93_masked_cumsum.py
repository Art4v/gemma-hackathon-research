"""Masked cumulative sum: cumsum(x * mask, axis=dim)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': '93_masked_cumsum',
    'batch_size': 4096,
    'seq_len': 16384,
    'dim': 1,
}


def create_inputs(dtype=jnp.float32):
    k1, k2 = jax.random.split(jax.random.key(0))
    shape = (CONFIG['batch_size'], CONFIG['seq_len'])
    x = jax.random.uniform(k1, shape, dtype=dtype)
    mask = jax.random.randint(k2, shape, 0, 2).astype(jnp.bool_)
    return x, mask


def workload(x, mask):
    return jnp.cumsum(x * mask.astype(x.dtype), axis=CONFIG['dim'])
