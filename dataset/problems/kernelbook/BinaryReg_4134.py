"""BinaryReg: alpha / sum(clamp(abs(x - 0.5), min=0.01))."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'BinaryReg_4134',
    'batch': 16,
    'channels': 16,
    'height': 512,
    'width': 512,
    'alpha': 1.0,
    'center': 0.5,
    'clamp_min': 0.01,
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    shape = (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width'])
    x = jax.random.uniform(key, shape, dtype)
    return (x,)


def workload(x):
    diff = jnp.maximum(jnp.abs(x - CONFIG['center']), CONFIG['clamp_min'])
    return CONFIG['alpha'] * (1.0 / jnp.sum(diff))
