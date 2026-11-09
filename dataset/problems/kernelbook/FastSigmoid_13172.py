"""[bf16 variant] FastSigmoid: x / (|x| + 1)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'FastSigmoid_13172',
    'batch': 16,
    'channels': 64,
    'height': 256,
    'width': 256,
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    shape = (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width'])
    x = jax.random.uniform(key, shape, dtype)
    return (x,)


def workload(x):
    return x / (jnp.abs(x) + 1)
