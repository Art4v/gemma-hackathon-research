"""ELU activation function."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': '31_ELU',
    'batch_size': 4096,
    'dim': 16384,
    'alpha': 1.0,
}

def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    x = jax.random.normal(key, (CONFIG['batch_size'], CONFIG['dim']), dtype=dtype)
    return (x,)

def workload(x):
    return jax.nn.elu(x, alpha=CONFIG['alpha'])
