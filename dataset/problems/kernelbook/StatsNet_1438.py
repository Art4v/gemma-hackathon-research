"""StatsNet_1438: flatten spatial dims -> mean + std (ddof=1) over flattened axis -> stack at dim 1."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'StatsNet_1438',
    'batch': 16,
    'channels': 128,
    'height': 128,
    'width': 128,
    'ddof': 1,
    'stack_axis': 1,
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    shape = (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width'])
    x = jax.random.uniform(key, shape, dtype)
    return (x,)


def workload(x):
    b, c, h, w = x.shape
    flat = jnp.reshape(x, (b, c, h * w))
    mean = jnp.mean(flat, axis=2)
    std = jnp.std(flat, axis=2, ddof=CONFIG['ddof'])
    return jnp.stack((mean, std), axis=CONFIG['stack_axis'])
