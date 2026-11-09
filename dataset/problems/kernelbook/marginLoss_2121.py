"""marginLoss: sum(max(pos - neg + margin, 0))."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'marginLoss_2121',
    'shape': (16, 128, 128, 128),
}


def create_inputs(dtype=jnp.float32):
    k_pos, k_neg, k_margin = jax.random.split(jax.random.key(0), 3)
    shape = CONFIG['shape']
    pos = jax.random.uniform(k_pos, shape, dtype)
    neg = jax.random.uniform(k_neg, shape, dtype)
    margin = jax.random.uniform(k_margin, shape, dtype)
    return pos, neg, margin


def workload(pos, neg, margin):
    val = pos - neg + margin
    return jnp.sum(jnp.maximum(val, jnp.zeros_like(val)))
