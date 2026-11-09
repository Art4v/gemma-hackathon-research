"""FM_2350: FM cross term, 0.5 * sum_e((sum_f X)^2 - sum_f X^2) on a 4D input."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'FM_2350',
    'batch': 32,
    'fields': 128,
    'embed': 128,
    'dim': 128,
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    x = jax.random.uniform(
        key,
        (CONFIG['batch'], CONFIG['fields'], CONFIG['embed'], CONFIG['dim']),
        dtype=dtype,
    )
    return (x,)


def workload(x):
    square_of_sum = jnp.square(jnp.sum(x, axis=1, keepdims=True))
    sum_of_square = jnp.sum(x * x, axis=1, keepdims=True)
    cross_term = square_of_sum - sum_of_square
    return 0.5 * jnp.sum(cross_term, axis=2)
