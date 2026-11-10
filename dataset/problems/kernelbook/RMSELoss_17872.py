"""RMSELoss: sqrt(mean((x - y)^2)) over all elements."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'RMSELoss_17872',
    'batch': 16,
    'channels': 32,
    'height': 256,
    'width': 256,
}


def create_inputs(dtype=jnp.float32):
    shape = (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width'])
    kx, ky = jax.random.split(jax.random.key(0))
    x = jax.random.uniform(kx, shape, dtype)
    y = jax.random.uniform(ky, shape, dtype)
    return x, y


def workload(x, y):
    diff = x - y
    mse = jnp.mean(jnp.square(diff), dtype=jnp.float32)
    return jnp.sqrt(mse).astype(x.dtype)
