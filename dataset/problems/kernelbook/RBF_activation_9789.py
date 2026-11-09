"""RBF activation: per-feature Gaussian pdf exp(-(x-c)^2/(2*s2)) / sqrt(2*pi*s2), s2 = exp(log_sigma2)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'RBF_activation_9789',
    'batch': 8,
    'dim1': 64,
    'dim2': 128,
    'input_features': 1024,
    'center_init': 1.0,
    'log_sigma2_init': -0.2,
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    shape = (CONFIG['batch'], CONFIG['dim1'], CONFIG['dim2'], CONFIG['input_features'])
    x = jax.random.uniform(key, shape, dtype)
    centers = jnp.full((CONFIG['input_features'],), CONFIG['center_init'], dtype)
    log_sigma2 = jnp.full((CONFIG['input_features'],), CONFIG['log_sigma2_init'], dtype)
    return x, centers, log_sigma2


def workload(x, centers, log_sigma2):
    s2 = jnp.exp(log_sigma2)
    diff = x - centers
    return 1 / jnp.sqrt(2 * jnp.pi * s2) * jnp.exp(-(diff * diff) / (2 * s2))
