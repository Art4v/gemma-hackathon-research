"""GELU (tanh approximation): 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'GELU_4334',
    'shape': (8, 128, 128, 512),
    'cubic_coeff': 0.044715,
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    x = jax.random.uniform(key, CONFIG['shape'], dtype)
    return (x,)


def workload(x):
    gelu_pow = x + CONFIG['cubic_coeff'] * x ** 3
    gelu_tanh = jnp.tanh((2.0 / jnp.pi) ** 0.5 * gelu_pow)
    return 0.5 * x * (1.0 + gelu_tanh)
