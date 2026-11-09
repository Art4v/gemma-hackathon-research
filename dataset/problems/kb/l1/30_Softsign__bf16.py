"""[bf16 variant] Softsign activation."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': '30_Softsign',
    'batch_size': 4096,
    'dim': 16384,
    'scale': 1.0,
}

def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    shape = (CONFIG['batch_size'], CONFIG['dim'])
    x = jax.random.uniform(key, shape, dtype=dtype)
    return (x,)

def workload(x):
    return x / (CONFIG['scale'] + jnp.abs(x))
