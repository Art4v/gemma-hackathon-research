"""[bf16 variant] Cumulative sum (prefix scan) along dim 1: torch.cumsum(x, dim=1)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': '89_cumsum',
    'batch_size': 4096,
    'seq_len': 16384,
    'dim': 1,
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    x = jax.random.uniform(key, (CONFIG['batch_size'], CONFIG['seq_len']), dtype=dtype)
    return (x,)


def workload(x):
    return jnp.cumsum(x, axis=CONFIG['dim'])
