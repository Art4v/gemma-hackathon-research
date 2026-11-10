"""co_peak_loss: mean(max(0,-x) + log(exp(-max(0,-x)) + exp(-x - max(0,-x)))), i.e. mean softplus(-x)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'co_peak_loss_6445',
    'batch': 16,
    'channels': 64,
    'height': 256,
    'width': 256,
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    shape = (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width'])
    co_peak_value = jax.random.uniform(key, shape, dtype)
    return (co_peak_value,)


def workload(co_peak_value):
    a = -1 * co_peak_value
    b = jnp.maximum(jnp.zeros_like(co_peak_value), a)
    t = b + jnp.log(jnp.exp(-b) + jnp.exp(a - b))
    return jnp.mean(t)
