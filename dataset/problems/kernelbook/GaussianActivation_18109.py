"""[bf16 variant] GaussianActivation: clamp params -> piecewise asymmetric gaussian (where x < mu)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'GaussianActivation_18109',
    'shape': (32, 64, 128, 128),
    'a_init': 4.0,
    'mu_init': 4.0,
    'gamma_l_init': 4.0,
    'gamma_r_init': 4.0,
    'a_range': (1.01, 6.0),
    'mu_range': (0.1, 3.0),
    'gamma_range': (0.5, 2.0),
}


def create_inputs(dtype=jnp.bfloat16):
    x = jax.random.uniform(jax.random.key(0), CONFIG['shape'], dtype)
    a = jnp.asarray(CONFIG['a_init'], dtype)
    mu = jnp.asarray(CONFIG['mu_init'], dtype)
    gamma_l = jnp.asarray(CONFIG['gamma_l_init'], dtype)
    gamma_r = jnp.asarray(CONFIG['gamma_r_init'], dtype)
    return x, a, mu, gamma_l, gamma_r


def workload(x, a, mu, gamma_l, gamma_r):
    a = jnp.clip(a, *CONFIG['a_range'])
    mu = jnp.clip(mu, *CONFIG['mu_range'])
    gamma_l = jnp.clip(gamma_l, *CONFIG['gamma_range'])
    gamma_r = jnp.clip(gamma_r, *CONFIG['gamma_range'])
    d2 = (x - mu) ** 2
    left = a * jnp.exp(-gamma_l * d2)
    right = 1 + (a - 1) * jnp.exp(-gamma_r * d2)
    return jnp.where(x < mu, left, right)
