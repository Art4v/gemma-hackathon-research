"""[m1 M2_second_output variant of kb/l1/90_cumprod] Cumulative product along dim 1, also returning each row's sum of the cumulative products (the expected survival length when x holds per-step survival probabilities)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': '90_cumprod_m1_aux',
    'batch_size': 4096,
    'length': 8192,
    'dim': 1,
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    x = jax.random.uniform(key, (CONFIG['batch_size'], CONFIG['length']), dtype=dtype)
    return (x,)


def workload(x):
    dim = CONFIG['dim']
    cp = jnp.cumprod(x, axis=dim)
    return cp, jnp.sum(cp, axis=dim)
