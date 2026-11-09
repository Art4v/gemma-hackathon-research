"""GELU (tanh approximation): 0.5*x*(1+tanh(sqrt(2/pi)*(x+0.044715*x^3)))."""
import math

import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'GELU__7911',
    'shape': (16, 128, 128, 128),
    'sqrt_2_over_pi': math.sqrt(2 / math.pi),
    'cubic_coeff': 0.044715,
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    x = jax.random.uniform(key, CONFIG['shape'], dtype)
    return (x,)


def workload(x):
    inner = CONFIG['sqrt_2_over_pi'] * (x + CONFIG['cubic_coeff'] * jnp.power(x, 3))
    return 0.5 * x * (1 + jnp.tanh(inner))
