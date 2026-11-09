"""[bf16 variant] QNetwork: Linear -> ReLU -> Linear -> ReLU -> Linear over the last axis."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'QNetwork_9978',
    'batch': 64,
    'd1': 64,
    'd2': 64,
    'state_size': 128,
    'fc1_units': 256,
    'fc2_units': 256,
    'action_size': 128,
}


def create_inputs(dtype=jnp.bfloat16):
    c = CONFIG
    keys = jax.random.split(jax.random.key(0), 7)
    x = jax.random.uniform(keys[0], (c['batch'], c['d1'], c['d2'], c['state_size']), dtype)
    w1 = jax.random.normal(keys[1], (c['state_size'], c['fc1_units']), dtype) * 0.02
    b1 = jax.random.normal(keys[2], (c['fc1_units'],), dtype) * 0.02
    w2 = jax.random.normal(keys[3], (c['fc1_units'], c['fc2_units']), dtype) * 0.02
    b2 = jax.random.normal(keys[4], (c['fc2_units'],), dtype) * 0.02
    w3 = jax.random.normal(keys[5], (c['fc2_units'], c['action_size']), dtype) * 0.02
    b3 = jax.random.normal(keys[6], (c['action_size'],), dtype) * 0.02
    return x, w1, b1, w2, b2, w3, b3


def workload(x, w1, b1, w2, b2, w3, b3):
    h = jax.nn.relu(x @ w1 + b1)
    h = jax.nn.relu(h @ w2 + b2)
    return h @ w3 + b3
