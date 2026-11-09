"""[bf16 variant] Softmax activation along the feature dimension."""

import jax
import jax.numpy as jnp

CONFIG = {
    'name': '23_Softmax',
    'batch_size': 4096,
    'dim': 32768,
    'axis': 1,
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    x = jax.random.uniform(key, (CONFIG['batch_size'], CONFIG['dim']), dtype=dtype)
    return (x,)


def workload(x):
    return jax.nn.softmax(x, axis=CONFIG['axis'])
