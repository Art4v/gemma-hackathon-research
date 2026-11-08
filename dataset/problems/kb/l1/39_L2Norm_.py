"""L2 normalization along axis 1."""

import jax
import jax.numpy as jnp

CONFIG = {
    'name': '39_L2Norm_',
    'batch_size': 8192,
    'dim': 16384,
    'norm_axis': 1,
}

def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    x = jax.random.uniform(key, (CONFIG['batch_size'], CONFIG['dim']), dtype=dtype)
    return (x,)

def workload(x):
    norm = jnp.linalg.norm(x, ord=2, axis=CONFIG['norm_axis'], keepdims=True)
    return x / norm
