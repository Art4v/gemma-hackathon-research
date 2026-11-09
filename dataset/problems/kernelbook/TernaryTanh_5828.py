"""[bf16 variant] TernaryTanh: 0.5*tanh(2*beta*x - beta) - 0.5*tanh(-2*beta*x - beta) (beta >= 1 branch)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'TernaryTanh_5828',
    'batch': 8,
    'channels': 32,
    'height': 512,
    'width': 512,
    'beta': 2.0,
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    shape = (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width'])
    x = jax.random.uniform(key, shape, dtype)
    return (x,)


def workload(x):
    # ponytail: only the beta >= 1 branch, which is the one the default beta=2.0 selects.
    beta = CONFIG['beta']
    y = jnp.tanh(x * beta * 2.0 - beta) * 0.5
    y = y + (-jnp.tanh(-x * beta * 2.0 - beta) * 0.5)
    return y
