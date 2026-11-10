"""[bf16 variant] Charbonnier loss: subtract, square, add eps, sqrt, sum, divide by element count."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'CharbonnierLoss_7813',
    'batch': 8,
    'channels': 128,
    'height': 128,
    'width': 256,
    'eps': 1e-06,
}


def create_inputs(dtype=jnp.bfloat16):
    shape = (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width'])
    key = jax.random.key(0)
    k_x, k_y = jax.random.split(key)
    x = jax.random.uniform(k_x, shape, dtype)
    y = jax.random.uniform(k_y, shape, dtype)
    return x, y


def workload(x, y):
    b, c, h, w = y.shape
    diff = x - y
    loss = jnp.sum(jnp.sqrt(diff * diff + CONFIG['eps']))
    return loss / (c * b * h * w)
