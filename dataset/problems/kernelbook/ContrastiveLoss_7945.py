"""ContrastiveLoss: label*0.25*(1-ew)^2 + (1-label)*where(ew<m,0,ew)^2, summed."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'ContrastiveLoss_7945',
    'shape': (8, 64, 256, 256),
    'pos_scale': 0.25,
}


def create_inputs(dtype=jnp.float32):
    k_ew, k_label, k_m = jax.random.split(jax.random.key(0), 3)
    shape = CONFIG['shape']
    ew = jax.random.uniform(k_ew, shape, dtype)
    label = jax.random.uniform(k_label, shape, dtype)
    m = jax.random.uniform(k_m, shape, dtype)
    return ew, label, m


def workload(ew, label, m):
    l_1 = CONFIG['pos_scale'] * (1.0 - ew) * (1.0 - ew)
    masked = jnp.where(ew < m, jnp.zeros_like(ew), ew)
    l_0 = masked * masked
    loss = label * l_1 + (1.0 - label) * l_0
    return jnp.sum(loss)
