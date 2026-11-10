"""[bf16 variant] GRelu: relu (or leaky_relu) -> optional subtract -> optional upper clamp."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'GRelu_12489',
    'batch': 8,
    'channels': 128,
    'height': 128,
    'width': 256,
    'leak': 0.0,
    'max': jnp.inf,
    'sub': 0.0,
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    shape = (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width'])
    x = jax.random.normal(key, shape, dtype)
    return (x,)


def workload(x):
    leak = CONFIG['leak']
    max_value = CONFIG['max']
    sub = CONFIG['sub']
    if leak:
        x = jnp.where(x >= 0, x, x * leak)
    else:
        x = jnp.maximum(x, jnp.zeros_like(x))
    if sub:
        x = x - sub
    if max_value:
        x = jnp.minimum(x, jnp.full_like(x, max_value))
    return x
