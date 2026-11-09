"""[bf16 variant] LeakyReLU activation."""

import jax
import jax.numpy as jnp

CONFIG = {
    'name': '20_LeakyReLU',
    'batch_size': 4096,
    'dim': 16384,
    'negative_slope': 0.01,
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    x = jax.random.normal(key, (CONFIG['batch_size'], CONFIG['dim']), dtype=dtype)
    return (x,)


def workload(x):
    return jax.nn.leaky_relu(x, negative_slope=CONFIG['negative_slope'])
