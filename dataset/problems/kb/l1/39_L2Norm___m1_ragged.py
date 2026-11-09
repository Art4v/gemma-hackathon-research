"""[m1 M3_ragged variant of kb/l1/39_L2Norm_] Row-wise L2 normalization y = x / ||x||_2, with the feature dim at 15000 instead of 16384 so it is not a multiple of 128."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': '39_L2Norm__m1_ragged',
    'batch_size': 2048,
    'dim': 15000,
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    x = jax.random.uniform(key, (CONFIG['batch_size'], CONFIG['dim']), dtype)
    return (x,)


def workload(x):
    sq = jnp.sum(x * x, axis=1, keepdims=True)
    norm = jnp.sqrt(sq)
    return x / norm
