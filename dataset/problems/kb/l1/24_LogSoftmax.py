"""LogSoftmax activation along axis 1."""

import jax
import jax.numpy as jnp

CONFIG = {
    'name': '24_LogSoftmax',
    'batch_size': 2048,
    'dim': 32768,
    'axis': 1,
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    x = jax.random.uniform(key, (CONFIG['batch_size'], CONFIG['dim']), dtype=dtype)
    return (x,)


def workload(x):
    return jax.nn.log_softmax(x, axis=CONFIG['axis'])
