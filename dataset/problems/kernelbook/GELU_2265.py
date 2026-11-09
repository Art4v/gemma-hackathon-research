"""[bf16 variant] GELU (tanh approximation): 0.5*x*(1+tanh(sqrt(2/pi)*(x+0.044715*x^3)))."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'GELU_2265',
    'shape': (32, 128, 128, 128),
    'cubic_coeff': 0.044715,
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    x = jax.random.uniform(key, CONFIG['shape'], dtype=dtype)
    return (x,)


def workload(x):
    c = jnp.sqrt(2.0 / jnp.pi).astype(x.dtype)
    inner = c * (x + CONFIG['cubic_coeff'] * x ** 3)
    return 0.5 * x * (1.0 + jnp.tanh(inner))
