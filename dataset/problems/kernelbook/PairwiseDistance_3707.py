"""[bf16 variant] PairwiseDistance_3707: abs difference -> square -> sum over dim 1 -> add smooth -> sqrt."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'PairwiseDistance_3707',
    'batch': 16,
    'channels': 128,
    'height': 128,
    'width': 128,
    'sum_axis': 1,
    'smooth': 0.0001,
}


def create_inputs(dtype=jnp.bfloat16):
    shape = (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width'])
    k1, k2 = jax.random.split(jax.random.key(0))
    x1 = jax.random.uniform(k1, shape, dtype)
    x2 = jax.random.uniform(k2, shape, dtype)
    return x1, x2


def workload(x1, x2):
    assert x1.shape == x2.shape
    diff = jnp.abs(x1 - x2)
    out = jnp.sum(jnp.square(diff), axis=CONFIG['sum_axis'])
    return jnp.sqrt(out + CONFIG['smooth'])
