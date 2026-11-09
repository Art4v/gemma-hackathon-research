"""GaussActivation: clamp params, piecewise Gaussian activation split at mu."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'GaussActivation_6544',
    'shape': (16, 64, 256, 256),
    'a': 4.0,
    'mu': 4.0,
    'sigma1': 4.0,
    'sigma2': 4.0,
    'a_min': 1.01, 'a_max': 6.0,
    'mu_min': 0.1, 'mu_max': 3.0,
    'sigma_min': 0.5, 'sigma_max': 2.0,
}


def create_inputs(dtype=jnp.float32):
    x = jax.random.uniform(jax.random.key(0), CONFIG['shape'], dtype)
    a = jnp.asarray(CONFIG['a'], dtype)
    mu = jnp.asarray(CONFIG['mu'], dtype)
    sigma1 = jnp.asarray(CONFIG['sigma1'], dtype)
    sigma2 = jnp.asarray(CONFIG['sigma2'], dtype)
    return x, a, mu, sigma1, sigma2


def workload(x, a, mu, sigma1, sigma2):
    a = jnp.clip(a, CONFIG['a_min'], CONFIG['a_max'])
    mu = jnp.clip(mu, CONFIG['mu_min'], CONFIG['mu_max'])
    sigma1 = jnp.clip(sigma1, CONFIG['sigma_min'], CONFIG['sigma_max'])
    sigma2 = jnp.clip(sigma2, CONFIG['sigma_min'], CONFIG['sigma_max'])
    d2 = (x - mu) ** 2
    left = a * jnp.exp(-sigma1 * d2)
    right = 1 + (a - 1) * jnp.exp(-sigma2 * d2)
    return jnp.where(x < mu, left, right)
