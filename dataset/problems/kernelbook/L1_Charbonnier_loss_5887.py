"""L1 Charbonnier loss: sum(sqrt((X - Y)^2 + eps))."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'L1_Charbonnier_loss_5887',
    'batch': 8,
    'channels': 64,
    'height': 256,
    'width': 256,
    'eps': 1e-06,
}


def create_inputs(dtype=jnp.float32):
    shape = (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width'])
    kx, ky = jax.random.split(jax.random.key(0))
    x = jax.random.uniform(kx, shape, dtype)
    y = jax.random.uniform(ky, shape, dtype)
    return x, y


def workload(x, y):
    diff = x - y
    error = jnp.sqrt(diff * diff + CONFIG['eps'])
    return jnp.sum(error)
