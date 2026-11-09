"""Beta2: sigmoid mean -> Beta(alpha, beta) parameters from a fixed log-std."""
import math

import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'Beta2_998',
    'batch': 16,
    'dim1': 32,
    'dim2': 64,
    'action_dim': 1024,
    'init_std': 0.25,
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    shape = (CONFIG['batch'], CONFIG['dim1'], CONFIG['dim2'], CONFIG['action_dim'])
    x = jax.random.uniform(key, shape, dtype)
    logstd = jnp.full((1, CONFIG['action_dim']), math.log(CONFIG['init_std']), dtype)
    return x, logstd


def workload(x, logstd):
    mean = jax.nn.sigmoid(x)
    var = jnp.exp(logstd) ** 2
    alpha = (1 - mean) / var * mean ** 2 - mean
    beta = (1 - mean) / var * mean - 1 - alpha
    return alpha, beta
