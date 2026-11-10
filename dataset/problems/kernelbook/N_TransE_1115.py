"""N_TransE score: translate (e1 + r - e2), vector p-norm over axis 1, negate."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'N_TransE_1115',
    'batch': 8,
    'channels': 128,
    'height': 128,
    'width': 128,
    'p': 4,
    'norm_axis': 1,
}


def create_inputs(dtype=jnp.float32):
    shape = (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width'])
    key = jax.random.key(0)
    k_e1, k_r, k_e2 = jax.random.split(key, 3)
    e1 = jax.random.uniform(k_e1, shape, dtype)
    r = jax.random.uniform(k_r, shape, dtype)
    e2 = jax.random.uniform(k_e2, shape, dtype)
    return e1, r, e2


def workload(e1, r, e2):
    p = CONFIG['p']
    diff = e1 + r - e2
    norm = jnp.sum(jnp.abs(diff) ** p, axis=CONFIG['norm_axis']) ** (1.0 / p)
    return -norm
