"""Element-wise clipping: minimum(x, max_value) followed by maximum(_, min_value)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'Clipping_14540',
    'shape': (8, 128, 128, 128),
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    k_x, k_max, k_min = jax.random.split(key, 3)
    shape = CONFIG['shape']
    x = jax.random.uniform(k_x, shape, dtype)
    max_value = jax.random.uniform(k_max, shape, dtype)
    min_value = jax.random.uniform(k_min, shape, dtype)
    return x, max_value, min_value


def workload(x, max_value, min_value):
    x = jnp.minimum(x, max_value)
    x = jnp.maximum(x, min_value)
    return x
