"""[bf16 variant] Cumulative product (cumprod) along dim 1."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': '90_cumprod',
    'batch_size': 4096,
    'length': 8192,
    'dim': 1,
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    x = jax.random.uniform(key, (CONFIG['batch_size'], CONFIG['length']), dtype=dtype)
    return (x,)


def workload(x):
    return jnp.cumprod(x, axis=CONFIG['dim'])
