"""RMSE loss: sqrt(mean((x - y) ** 2))."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'RMSELoss_1964',
    'batch': 16,
    'channels': 32,
    'height': 256,
    'width': 256,
}


def create_inputs(dtype=jnp.float32):
    kx, ky = jax.random.split(jax.random.key(0))
    shape = (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width'])
    x = jax.random.uniform(kx, shape, dtype)
    y = jax.random.uniform(ky, shape, dtype)
    return x, y


def workload(x, y):
    return jnp.sqrt(jnp.mean(jnp.square(x - y)))
