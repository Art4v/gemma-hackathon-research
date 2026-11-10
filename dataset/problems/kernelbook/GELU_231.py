"""[bf16 variant] GELU tanh approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'GELU_231',
    'shape': (8, 128, 128, 128),
    'outer_scale': 0.5,
    'offset': 1.0,
    'sqrt_2_over_pi': 0.7978845608028654,
    'cubic_coeff': 0.044715,
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    x = jax.random.uniform(key, CONFIG['shape'], dtype)
    return (x,)


def workload(x):
    inner = CONFIG['sqrt_2_over_pi'] * (x + CONFIG['cubic_coeff'] * jnp.power(x, 3))
    return CONFIG['outer_scale'] * x * (CONFIG['offset'] + jnp.tanh(inner))
