"""BetaMish: x * tanh(log((1 + exp(x)) ** beta)), elementwise."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'BetaMish_3429',
    'batch': 16,
    'channels': 64,
    'height': 256,
    'width': 256,
    'beta': 1.5,
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    shape = (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width'])
    x = jax.random.uniform(key, shape, dtype)
    return (x,)


def workload(x):
    beta = CONFIG['beta']
    return x * jnp.tanh(jnp.log(jnp.power(1 + jnp.exp(x), beta)))
