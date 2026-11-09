"""WeightedLoss: mean(square((y - Y) / divisor) * w)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'WeightedLoss_12047',
    'batch': 8,
    'channels': 64,
    'height': 256,
    'width': 256,
    'divisor': 5.0,
}


def create_inputs(dtype=jnp.float32):
    shape = (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width'])
    k1, k2, k3 = jax.random.split(jax.random.key(0), 3)
    y = jax.random.uniform(k1, shape, dtype)
    Y = jax.random.uniform(k2, shape, dtype)
    w = jax.random.uniform(k3, shape, dtype)
    return y, Y, w


def workload(y, Y, w):
    diff = (y - Y) / CONFIG['divisor']
    return jnp.mean(jnp.square(diff) * w)
