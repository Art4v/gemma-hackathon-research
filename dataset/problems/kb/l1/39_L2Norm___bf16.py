"""[bf16 variant] L2 normalization along dim=1: y = x / ||x||_2 (sum-of-squares reduce, sqrt, divide)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': '39_L2Norm_',
    'batch_size': 2048,
    'dim': 16384,
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    x = jax.random.uniform(key, (CONFIG['batch_size'], CONFIG['dim']), dtype)
    return (x,)


def workload(x):
    sq = jnp.sum(x * x, axis=1, keepdims=True)
    norm = jnp.sqrt(sq)
    return x / norm
