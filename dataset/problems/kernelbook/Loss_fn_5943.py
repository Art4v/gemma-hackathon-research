"""Charbonnier loss: mean(sqrt((ip - target)^2 + eps^2))."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'Loss_fn_5943',
    'batch': 16,
    'channels': 64,
    'height': 128,
    'width': 128,
    'eps': 0.001,
}


def create_inputs(dtype=jnp.float32):
    shape = (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width'])
    k1, k2 = jax.random.split(jax.random.key(0))
    ip = jax.random.uniform(k1, shape, dtype)
    target = jax.random.uniform(k2, shape, dtype)
    return ip, target


def workload(ip, target):
    eps = CONFIG['eps']
    diff = ip - target
    return jnp.mean(jnp.sqrt(diff * diff + eps * eps))
