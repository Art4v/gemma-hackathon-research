"""Hswish: x * relu6(x + 3) * (1/6) (FloatFunctional add_scalar -> ReLU6 -> mul_scalar -> mul)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'Hswish_8080',
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
    x = jax.random.uniform(key, shape, dtype=dtype)
    return (x,)


def workload(x):
    hsig = jnp.clip(x + CONFIG['add_scalar'], 0.0, CONFIG['relu6_max']) * CONFIG['mul_scalar']
    return x * hsig
