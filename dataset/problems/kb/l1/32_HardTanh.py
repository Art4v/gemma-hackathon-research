"""HardTanh activation."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': '32_HardTanh',
    'batch_size': 4096,
    'dim': 16384,
    'min_val': -1.0,
    'max_val': 1.0,
}

def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    x = jax.random.normal(key, (CONFIG['batch_size'], CONFIG['dim']), dtype=dtype)
    return (x,)

def workload(x):
    return jnp.clip(x, CONFIG['min_val'], CONFIG['max_val'])
