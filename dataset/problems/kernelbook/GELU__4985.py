"""GELU tanh approximation: x^3 -> scaled sum -> tanh -> 0.5 * x * (1 + tanh)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'GELU__4985',
    'batch': 16,
    'dim1': 128,
    'dim2': 128,
    'dim3': 128,
    'outer_scale': 0.5,
    'offset': 1.0,
    'sqrt_2_over_pi': 0.7978845608028654,
    'cubic_coeff': 0.044715,
    'cubic_power': 3,
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    shape = (CONFIG['batch'], CONFIG['dim1'], CONFIG['dim2'], CONFIG['dim3'])
    x = jax.random.uniform(key, shape, dtype)
    return (x,)


def workload(x):
    inner = CONFIG['sqrt_2_over_pi'] * (
        x + CONFIG['cubic_coeff'] * jnp.power(x, CONFIG['cubic_power'])
    )
    return CONFIG['outer_scale'] * x * (CONFIG['offset'] + jnp.tanh(inner))
