"""[bf16 variant] SoftL1: abs difference, subtract eps, clamp to [0, 100], plus mean of the abs difference."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'SoftL1_14509',
    'shape': (8, 128, 128, 128),
    'eps': 0.0,
    'clamp_min': 0.0,
    'clamp_max': 100.0,
}


def create_inputs(dtype=jnp.bfloat16):
    k_input, k_target = jax.random.split(jax.random.key(0))
    input_ = jax.random.uniform(k_input, CONFIG['shape'], dtype)
    target = jax.random.uniform(k_target, CONFIG['shape'], dtype)
    return input_, target


def workload(input_, target):
    l1 = jnp.abs(input_ - target)
    ret = l1 - CONFIG['eps']
    ret = jnp.clip(ret, CONFIG['clamp_min'], CONFIG['clamp_max'])
    return ret, jnp.mean(jax.lax.stop_gradient(l1))
