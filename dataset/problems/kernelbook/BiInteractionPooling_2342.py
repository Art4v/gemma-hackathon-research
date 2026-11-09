"""[bf16 variant] BiInteractionPooling: 0.5 * (sum_f(x)^2 - sum_f(x*x)) over dim 1, keepdim."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'BiInteractionPooling_2342',
    'batch': 32,
    'fields': 64,
    'height': 128,
    'width': 128,
    'reduce_axis': 1,
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    shape = (CONFIG['batch'], CONFIG['fields'], CONFIG['height'], CONFIG['width'])
    x = jax.random.uniform(key, shape, dtype=dtype)
    return (x,)


def workload(x):
    axis = CONFIG['reduce_axis']
    square_of_sum = jnp.square(jnp.sum(x, axis=axis, keepdims=True))
    sum_of_square = jnp.sum(x * x, axis=axis, keepdims=True)
    return 0.5 * (square_of_sum - sum_of_square)
