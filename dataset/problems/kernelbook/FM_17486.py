"""[bf16 variant] FM_17486: Factorization Machine pairwise interaction: 0.5 * sum_dim2((sum_dim1 x)^2 - sum_dim1 x^2)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'FM_17486',
    'batch': 256,
    'fields': 16,
    'dim2': 128,
    'embed': 128,
    'field_axis': 1,
    'reduce_axis': 2,
    'scale': 0.5,
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    x = jax.random.uniform(
        key,
        (CONFIG['batch'], CONFIG['fields'], CONFIG['dim2'], CONFIG['embed']),
        dtype=dtype,
    )
    return (x,)


def workload(x):
    s = jnp.sum(x, axis=CONFIG['field_axis'], keepdims=True)
    square_of_sum = s * s
    sum_of_square = jnp.sum(x * x, axis=CONFIG['field_axis'], keepdims=True)
    cross = square_of_sum - sum_of_square
    return CONFIG['scale'] * jnp.sum(cross, axis=CONFIG['reduce_axis'])
