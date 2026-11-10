"""[bf16 variant] LogSumPenalty: abs -> add epsilon -> log -> sum."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'LogSumPenalty_10067',
    'shape': (16, 128, 128, 128),
    'epsilon': 1.0,
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    x = jax.random.uniform(key, CONFIG['shape'], dtype)
    return (x,)


def workload(x):
    return jnp.sum(jnp.log(jnp.abs(x) + CONFIG['epsilon']))
