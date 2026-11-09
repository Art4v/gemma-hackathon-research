"""L1 Charbonnier loss: mean(sqrt((X - Y)^2 + eps^2))."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'L1_Charbonnier_loss_6002',
    'batch': 8,
    'channels': 64,
    'height': 256,
    'width': 256,
    'eps': 0.001,
}


def create_inputs(dtype=jnp.float32):
    kx, ky = jax.random.split(jax.random.key(0))
    shape = (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width'])
    x = jax.random.uniform(kx, shape, dtype)
    y = jax.random.uniform(ky, shape, dtype)
    return x, y


def workload(x, y):
    eps = CONFIG['eps']
    diff = x - y
    error = jnp.sqrt(diff * diff + eps * eps)
    return jnp.mean(error)
