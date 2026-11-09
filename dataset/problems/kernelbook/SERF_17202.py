"""SERF activation: x * erf(log1p(exp(min(x, thresh))))."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'SERF_17202',
    'batch': 32,
    'channels': 128,
    'height': 128,
    'width': 128,
    'thresh': 50.0,
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    shape = (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width'])
    x = jax.random.uniform(key, shape, dtype=dtype)
    return (x,)


def workload(x):
    clamped = jnp.minimum(x, CONFIG['thresh'])
    return x * jax.scipy.special.erf(jnp.log1p(jnp.exp(clamped)))
