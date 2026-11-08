"""L1 normalization along dimension 1."""

import jax
import jax.numpy as jnp

CONFIG = {
    'name': '38_L1Norm_',
    'batch_size': 8192,
    'dim': 8192,
    'axis': 1,
}

def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    shape = (CONFIG['batch_size'], CONFIG['dim'])
    x = jax.random.uniform(key, shape, dtype=dtype)
    return (x,)

def workload(x):
    mean_abs = jnp.mean(jnp.abs(x), axis=CONFIG['axis'], keepdims=True)
    return x / mean_abs
