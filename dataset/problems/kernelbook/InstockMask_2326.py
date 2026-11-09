"""InstockMask: round((instock_dph + eps) / (total_dph + eps)) >= min_ratio ? demand : -1."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'InstockMask_2326',
    'shape': (16, 128, 128, 128),
    'min_instock_ratio': 0.5,
    'eps_instock_dph': 0.001,
    'eps_total_dph': 0.001,
}


def create_inputs(dtype=jnp.float32):
    k1, k2, k3 = jax.random.split(jax.random.key(0), 3)
    shape = CONFIG['shape']
    demand = jax.random.uniform(k1, shape, dtype)
    total_dph = jax.random.uniform(k2, shape, dtype)
    instock_dph = jax.random.uniform(k3, shape, dtype)
    return demand, total_dph, instock_dph


def workload(demand, total_dph, instock_dph):
    total = total_dph + CONFIG['eps_total_dph']
    instock = instock_dph + CONFIG['eps_instock_dph']
    instock_rate = jnp.round(instock / total)
    return jnp.where(instock_rate >= CONFIG['min_instock_ratio'], demand, -jnp.ones_like(demand))
