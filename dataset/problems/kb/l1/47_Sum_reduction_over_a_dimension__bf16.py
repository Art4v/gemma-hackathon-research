"""[bf16 variant] Sum reduction over a dimension (keepdim=True)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': '47_Sum_reduction_over_a_dimension',
    'batch_size': 16,
    'dim1': 2048,
    'dim2': 2048,
    'reduce_dim': 1,
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    x = jax.random.uniform(
        key, (CONFIG['batch_size'], CONFIG['dim1'], CONFIG['dim2']), dtype=dtype)
    return (x,)


def workload(x):
    return jnp.sum(x, axis=CONFIG['reduce_dim'], keepdims=True)
