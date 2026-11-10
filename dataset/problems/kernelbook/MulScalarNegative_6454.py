"""MulScalarNegative: float-mode QuantStub -> multiply by negative scalar -> DeQuantStub (elementwise x * -0.3)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'MulScalarNegative_6454',
    'batch': 8,
    'channels': 128,
    'height': 256,
    'width': 256,
    'scalar': -0.3,
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    shape = (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width'])
    x = jax.random.uniform(key, shape, dtype)
    return (x,)


def workload(x):
    return x * CONFIG['scalar']
