"""[bf16 variant] GaussMembFunc: exp(-(x - mu)^2 / (2 * sigma^2)) with scalar mu, sigma."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'GaussMembFunc_4261',
    'batch': 8,
    'channels': 128,
    'height': 256,
    'width': 256,
    'mu': 4.0,
    'sigma': 4.0,
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    shape = (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width'])
    x = jax.random.uniform(key, shape, dtype)
    mu = jnp.asarray(CONFIG['mu'], dtype=dtype)
    sigma = jnp.asarray(CONFIG['sigma'], dtype=dtype)
    return x, mu, sigma


def workload(x, mu, sigma):
    return jnp.exp(-jnp.square(x - mu) / (2 * sigma ** 2))
