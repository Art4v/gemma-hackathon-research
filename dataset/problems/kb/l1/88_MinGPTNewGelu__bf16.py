"""[bf16 variant] MinGPT NewGELU activation function (tanh approximation)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': '88_MinGPTNewGelu',
    'batch_size': 8192,
    'dim': 8192,
    'scale': 0.5,
    'cubic_coeff': 0.044715,
}

def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    x = jax.random.uniform(key, (CONFIG['batch_size'], CONFIG['dim']), dtype=dtype)
    return (x,)

def workload(x):
    sqrt_2_over_pi = jnp.sqrt(2.0 / jnp.pi)
    return CONFIG['scale'] * x * (1.0 + jnp.tanh(sqrt_2_over_pi * (x + CONFIG['cubic_coeff'] * jnp.power(x, 3.0))))
