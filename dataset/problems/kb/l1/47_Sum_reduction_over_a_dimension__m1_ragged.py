"""[m1 M3_ragged variant of kb/l1/47_Sum_reduction_over_a_dimension] Sum reduction over dim 1 (keepdim=True), with the reduced axis at 1500 (a Whisper-encoder frame count) instead of 2048, so it is a multiple of neither 128 nor 8."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': '47_Sum_reduction_over_a_dimension_m1_ragged',
    'batch_size': 16,
    'dim1': 1500,
    'dim2': 2048,
    'reduce_dim': 1,
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    x = jax.random.uniform(
        key, (CONFIG['batch_size'], CONFIG['dim1'], CONFIG['dim2']), dtype=dtype)
    return (x,)


def workload(x):
    return jnp.sum(x, axis=CONFIG['reduce_dim'], keepdims=True)
