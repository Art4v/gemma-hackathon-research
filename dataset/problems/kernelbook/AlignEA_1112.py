"""[bf16 variant] AlignEA forward: sum((e1 + r - e2)^2, axis=1)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'AlignEA_1112',
    'batch': 16,
    'channels': 128,
    'height': 128,
    'width': 256,
    'reduce_axis': 1,
}


def create_inputs(dtype=jnp.bfloat16):
    shape = (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width'])
    k1, k2, k3 = jax.random.split(jax.random.key(0), 3)
    e1 = jax.random.uniform(k1, shape, dtype)
    r = jax.random.uniform(k2, shape, dtype)
    e2 = jax.random.uniform(k3, shape, dtype)
    return e1, r, e2


def workload(e1, r, e2):
    d = e1 + r - e2
    return jnp.sum(d * d, axis=CONFIG['reduce_axis'])
