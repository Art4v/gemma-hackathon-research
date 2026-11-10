"""soft_L1: abs(input - target) - eps, clamped to [min, max] elementwise."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'soft_L1_15480',
    'batch': 8,
    'channels': 128,
    'height': 128,
    'width': 128,
    'eps': 0.0,
    'clamp_min': 0.0,
    'clamp_max': 100.0,
}


def create_inputs(dtype=jnp.float32):
    shape = (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width'])
    key = jax.random.key(0)
    k_input, k_target = jax.random.split(key)
    input_ = jax.random.uniform(k_input, shape, dtype)
    target = jax.random.uniform(k_target, shape, dtype)
    return input_, target


def workload(input_, target):
    ret = jnp.abs(input_ - target) - CONFIG['eps']
    return jnp.clip(ret, CONFIG['clamp_min'], CONFIG['clamp_max'])
