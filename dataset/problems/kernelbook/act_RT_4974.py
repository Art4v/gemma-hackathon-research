"""[bf16 variant] act_RT: elementwise (relu(x) + tanh(x)) / 2 on a 4D tensor."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'act_RT_4974',
    'batch': 16,
    'channels': 128,
    'height': 128,
    'width': 128,
    'divisor': 2.0,
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    shape = (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width'])
    x = jax.random.uniform(key, shape, dtype)
    return (x,)


def workload(x):
    return (jnp.maximum(x, 0.0) + jnp.tanh(x)) / CONFIG['divisor']
