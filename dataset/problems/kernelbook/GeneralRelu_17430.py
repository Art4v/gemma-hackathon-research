"""[bf16 variant] GeneralRelu: leaky_relu or relu, then optional subtract, then optional clamp_max."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'GeneralRelu_17430',
    'shape': (16, 128, 128, 128),
    'leak': None,
    'sub': None,
    'maxv': None,
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    x = jax.random.normal(key, CONFIG['shape'], dtype)
    return (x,)


def workload(x):
    leak, sub, maxv = CONFIG['leak'], CONFIG['sub'], CONFIG['maxv']
    if leak is not None:
        x = jnp.where(x > 0, x, x * leak)
    else:
        x = jnp.maximum(x, jnp.zeros_like(x))
    if sub is not None:
        x = x - sub
    if maxv is not None:
        x = jnp.minimum(x, jnp.full_like(x, maxv))
    return x
