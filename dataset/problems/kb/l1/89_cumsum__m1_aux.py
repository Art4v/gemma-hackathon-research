"""[m1 M2_second_output variant of kb/l1/89_cumsum] Cumulative sum along dim 1, also returning the per-row absmax of the cumsum output (a per-row quantization scale)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': '89_cumsum_m1_aux',
    'batch_size': 4096,
    'seq_len': 16384,
    'dim': 1,
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    x = jax.random.uniform(key, (CONFIG['batch_size'], CONFIG['seq_len']), dtype=dtype)
    return (x,)


def workload(x):
    dim = CONFIG['dim']
    cs = jnp.cumsum(x, axis=dim)
    return cs, jnp.max(jnp.abs(cs), axis=dim, keepdims=True)
