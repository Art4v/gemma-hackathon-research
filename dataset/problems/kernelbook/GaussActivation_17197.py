"""[bf16 variant] GaussActivation: clamp params, piecewise a*exp(-s1*(x-mu)^2) for x<mu, 1+(a-1)*exp(-s2*(x-mu)^2) for x>=mu."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'GaussActivation_17197',
    'shape': (32, 128, 128, 128),
    'a_init': 4.0,
    'mu_init': 4.0,
    'sigma1_init': 4.0,
    'sigma2_init': 4.0,
    'a_range': (1.01, 6.0),
    'mu_range': (0.1, 3.0),
    'sigma1_range': (1.0, 2.0),
    'sigma2_range': (1.0, 2.0),
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    x = jax.random.uniform(key, CONFIG['shape'], dtype)
    a = jnp.asarray(CONFIG['a_init'], dtype)
    mu = jnp.asarray(CONFIG['mu_init'], dtype)
    sigma1 = jnp.asarray(CONFIG['sigma1_init'], dtype)
    sigma2 = jnp.asarray(CONFIG['sigma2_init'], dtype)
    return x, a, mu, sigma1, sigma2


def workload(x, a, mu, sigma1, sigma2):
    a = jnp.clip(a, *CONFIG['a_range'])
    mu = jnp.clip(mu, *CONFIG['mu_range'])
    sigma1 = jnp.clip(sigma1, *CONFIG['sigma1_range'])
    sigma2 = jnp.clip(sigma2, *CONFIG['sigma2_range'])
    d2 = (x - mu) ** 2
    left = a * jnp.exp(-sigma1 * d2)
    right = 1 + (a - 1) * jnp.exp(-sigma2 * d2)
    zero = jnp.zeros_like(left)
    return jnp.where(x < mu, left, zero) + jnp.where(x >= mu, right, zero)
