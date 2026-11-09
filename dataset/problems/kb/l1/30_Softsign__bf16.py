"""[bf16 variant] Softsign activation: x / (1 + |x|)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': '30_Softsign',
    'batch_size': 4096,
    'dim': 16384,
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    x = jax.random.uniform(key, (CONFIG['batch_size'], CONFIG['dim']), dtype=dtype)
    return (x,)


def workload(x):
    return x / (1 + jnp.abs(x))
