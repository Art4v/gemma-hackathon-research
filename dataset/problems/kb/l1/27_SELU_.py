"""SELU activation."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': '27_SELU_',
    'batch_size': 2048,
    'dim': 32768,
}

def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    x = jax.random.normal(key, (CONFIG['batch_size'], CONFIG['dim']), dtype=dtype)
    return (x,)

def workload(x):
    return jax.nn.selu(x)
