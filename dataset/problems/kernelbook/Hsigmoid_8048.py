"""Hsigmoid: relu6(x + 3) * (1/6)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'Hsigmoid_8048',
    'batch': 16,
    'channels': 64,
    'height': 256,
    'width': 256,
    'add_scalar': 3.0,
    'relu6_max': 6.0,
    'mul_scalar': 1.0 / 6.0,
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    shape = (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width'])
    x = jax.random.uniform(key, shape, dtype)
    return (x,)


def workload(x):
    relu6 = jnp.clip(x + CONFIG['add_scalar'], 0.0, CONFIG['relu6_max'])
    return relu6 * CONFIG['mul_scalar']
