"""ReLU: y = max(x, 0) elementwise."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': '19_ReLU',
    'batch_size': 4096,
    'dim': 16384,
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    x = jax.random.normal(key, (CONFIG['batch_size'], CONFIG['dim']), dtype=dtype)
    return (x,)


def workload(x):
    return jnp.maximum(x, jnp.zeros((), dtype=x.dtype))
