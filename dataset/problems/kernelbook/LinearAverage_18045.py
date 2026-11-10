"""[bf16 variant] LinearAverage: out = (x @ memory.T) / T."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'LinearAverage_18045',
    'batch': 4096,
    'input_size': 2048,
    'output_size': 8192,
    'T': 0.05,
    'momentum': 0.5,
}


def create_inputs(dtype=jnp.bfloat16):
    kx, km = jax.random.split(jax.random.key(0))
    x = jax.random.uniform(kx, (CONFIG['batch'], CONFIG['input_size']), dtype)
    memory = jax.random.normal(km, (CONFIG['output_size'], CONFIG['input_size']), dtype) * 0.02
    return x, memory


def workload(x, memory):
    return jnp.matmul(x, memory.T) / CONFIG['T']
