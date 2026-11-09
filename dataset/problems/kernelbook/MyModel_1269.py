"""[bf16 variant] Linear -> ReLU -> Linear -> ReLU -> Linear MLP over the last axis."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'MyModel_1269',
    'batch': 64,
    'dim1': 64,
    'dim2': 64,
    'state_size': 128,
    'hidden1': 120,
    'hidden2': 84,
    'action_size': 128,
}


def create_inputs(dtype=jnp.bfloat16):
    keys = jax.random.split(jax.random.key(0), 7)
    c = CONFIG
    x = jax.random.uniform(keys[0], (c['batch'], c['dim1'], c['dim2'], c['state_size']), dtype)
    w1 = jax.random.normal(keys[1], (c['state_size'], c['hidden1']), dtype) * 0.02
    b1 = jax.random.normal(keys[2], (c['hidden1'],), dtype) * 0.02
    w2 = jax.random.normal(keys[3], (c['hidden1'], c['hidden2']), dtype) * 0.02
    b2 = jax.random.normal(keys[4], (c['hidden2'],), dtype) * 0.02
    w3 = jax.random.normal(keys[5], (c['hidden2'], c['action_size']), dtype) * 0.02
    b3 = jax.random.normal(keys[6], (c['action_size'],), dtype) * 0.02
    return x, w1, b1, w2, b2, w3, b3


def workload(x, w1, b1, w2, b2, w3, b3):
    h = jax.nn.relu(x @ w1 + b1)
    h = jax.nn.relu(h @ w2 + b2)
    return h @ w3 + b3
