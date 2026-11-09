"""[bf16 variant] MinGPT NewGELU: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))."""
import math

import jax
import jax.numpy as jnp

CONFIG = {
    'name': '88_MinGPTNewGelu',
    'batch_size': 8192,
    'dim': 8192,
    'coeff': 0.044715,
    'sqrt_2_over_pi': math.sqrt(2.0 / math.pi),
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    x = jax.random.uniform(key, (CONFIG['batch_size'], CONFIG['dim']), dtype=dtype)
    return (x,)


def workload(x):
    c = CONFIG['sqrt_2_over_pi']
    a = CONFIG['coeff']
    return 0.5 * x * (1.0 + jnp.tanh(c * (x + a * jnp.power(x, 3.0))))
