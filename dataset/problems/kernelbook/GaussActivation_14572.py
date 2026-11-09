"""[bf16 variant] GaussActivation: clamp params, then piecewise Gaussian (left: a*exp(-s1*d^2), right: 1+(a-1)*exp(-s2*d^2))."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'GaussActivation_14572',
    'shape': (8, 128, 256, 256),
    'a': 4.0,
    'mu': 4.0,
    'sigma1': 4.0,
    'sigma2': 4.0,
    'a_range': (1.01, 6.0),
    'mu_range': (0.1, 3.0),
    'sigma_range': (0.5, 2.0),
}


def create_inputs(dtype=jnp.bfloat16):
    x = jax.random.uniform(jax.random.key(0), CONFIG['shape'], dtype)
    a = jnp.full((), CONFIG['a'], dtype)
    mu = jnp.full((), CONFIG['mu'], dtype)
    sigma1 = jnp.full((), CONFIG['sigma1'], dtype)
    sigma2 = jnp.full((), CONFIG['sigma2'], dtype)
    return x, a, mu, sigma1, sigma2


def workload(x, a, mu, sigma1, sigma2):
    a = jnp.clip(a, *CONFIG['a_range'])
    mu = jnp.clip(mu, *CONFIG['mu_range'])
    sigma1 = jnp.clip(sigma1, *CONFIG['sigma_range'])
    sigma2 = jnp.clip(sigma2, *CONFIG['sigma_range'])
    d2 = (x - mu) ** 2
    left = a * jnp.exp(-sigma1 * d2)
    right = 1 + (a - 1) * jnp.exp(-sigma2 * d2)
    return jnp.where(x < mu, left, right)
