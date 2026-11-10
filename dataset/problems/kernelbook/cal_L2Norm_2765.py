"""cal_L2Norm: sqrt(sum(x * x, axis=1) + eps) over the channel axis."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'cal_L2Norm_2765',
    'batch': 16,
    'channels': 128,
    'height': 128,
    'width': 128,
    'axis': 1,
    'eps': 1e-10,
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    shape = (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width'])
    x = jax.random.uniform(key, shape, dtype)
    return (x,)


def workload(x):
    return jnp.sqrt(jnp.sum(x * x, axis=CONFIG['axis']) + CONFIG['eps'])
