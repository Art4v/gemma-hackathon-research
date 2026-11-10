"""BoundSoftmaxImpl: max along axis -> subtract -> exp -> sum along axis -> divide (softmax with max-shift)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'BoundSoftmaxImpl_7253',
    'batch': 8,
    'd1': 8,
    'd2': 128,
    'd3': 128,
    'd4': 128,
    'axis': 4,
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    shape = (CONFIG['batch'], CONFIG['d1'], CONFIG['d2'], CONFIG['d3'], CONFIG['d4'])
    x = jax.random.uniform(key, shape, dtype)
    return (x,)


def workload(x):
    axis = CONFIG['axis']
    max_x = jnp.max(x, axis=axis, keepdims=True)
    e = jnp.exp(x - max_x)
    s = jnp.sum(e, axis=axis, keepdims=True)
    return e / s
