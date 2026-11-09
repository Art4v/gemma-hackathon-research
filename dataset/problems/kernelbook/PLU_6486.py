"""[bf16 variant] PLU: w1 * max(x, 0) + w2 * min(x, 0)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'PLU_6486',
    'batch': 16,
    'channels': 64,
    'height': 256,
    'width': 256,
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    shape = (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width'])
    x = jax.random.normal(key, shape, dtype)
    w1 = jnp.ones((1,), dtype)
    w2 = jnp.ones((1,), dtype)
    return x, w1, w2


def workload(x, w1, w2):
    return w1 * jnp.maximum(x, 0) + w2 * jnp.minimum(x, 0)
