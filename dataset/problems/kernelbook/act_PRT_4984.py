"""[bf16 variant] ReLU + PReLU + Tanh averaged: (relu(x) + prelu(x) + tanh(x)) / 3."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'act_PRT_4984',
    'batch': 16,
    'channels': 64,
    'height': 256,
    'width': 256,
    'prelu_init': 0.25,
    'num_branches': 3,
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    shape = (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width'])
    # normal (not uniform [0,1)) so the negative PReLU branch is exercised
    x = jax.random.normal(key, shape, dtype)
    prelu_weight = jnp.full((1,), CONFIG['prelu_init'], dtype)
    return x, prelu_weight


def workload(x, prelu_weight):
    a = prelu_weight[0].astype(x.dtype)
    relu = jnp.maximum(x, 0)
    prelu = jnp.where(x >= 0, x, a * x)
    tanh = jnp.tanh(x)
    return (relu + prelu + tanh) / CONFIG['num_branches']
