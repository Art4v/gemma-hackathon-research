"""[bf16 variant] Fair loss: sum(c^2 * (|X-Y|/c - log(1 + |X-Y|/c)))."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'Fair_1796',
    'batch': 16,
    'channels': 32,
    'height': 256,
    'width': 256,
    'c': 1.0,
}


def create_inputs(dtype=jnp.bfloat16):
    kx, ky = jax.random.split(jax.random.key(0))
    shape = (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width'])
    x = jax.random.uniform(kx, shape, dtype)
    y = jax.random.uniform(ky, shape, dtype)
    return x, y


def workload(x, y):
    c = CONFIG['c']
    ra = jnp.abs(x - y)
    error = c ** 2 * (ra / c - jnp.log(1 + ra / c))
    return jnp.sum(error)
