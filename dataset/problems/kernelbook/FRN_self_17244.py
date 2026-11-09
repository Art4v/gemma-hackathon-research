"""Filter Response Normalization: x * rsqrt(mean(x^2 over HW) + |eps|), then gamma * x + beta."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'FRN_self_17244',
    'batch': 8,
    'num_features': 128,
    'height': 256,
    'width': 256,
    'eps': 1e-5,
}


def create_inputs(dtype=jnp.float32):
    c = CONFIG
    key = jax.random.key(0)
    x = jax.random.uniform(key, (c['batch'], c['num_features'], c['height'], c['width']), dtype)
    gamma = jnp.ones((1, c['num_features'], 1, 1), dtype)
    beta = jnp.zeros((1, c['num_features'], 1, 1), dtype)
    eps = jnp.full((1,), c['eps'], dtype)
    return x, gamma, beta, eps


def workload(x, gamma, beta, eps):
    nu2 = jnp.mean(jnp.square(x), axis=(2, 3), keepdims=True)
    y = x * jax.lax.rsqrt(nu2 + jnp.abs(eps))
    return gamma * y + beta
