"""[bf16 variant] Charbonnier loss: sum(sqrt((X - Y)^2 + eps))."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'Charbonnier_1797',
    'batch': 16,
    'channels': 32,
    'height': 256,
    'width': 256,
    'eps': 1e-06,
}


def create_inputs(dtype=jnp.bfloat16):
    kx, ky = jax.random.split(jax.random.key(0))
    shape = (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width'])
    x = jax.random.uniform(kx, shape, dtype)
    y = jax.random.uniform(ky, shape, dtype)
    return x, y


def workload(x, y):
    diff = x - y
    error = jnp.sqrt(diff * diff + CONFIG['eps'])
    return jnp.sum(error)
