"""QNetwork: Linear -> ReLU -> Linear -> ReLU -> Linear over the last axis."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'QNetwork_13927',
    'batch': 32,
    'dim1': 64,
    'dim2': 128,
    'num_states': 128,
    'hidden1': 100,
    'hidden2': 60,
    'num_actions': 128,
}


def create_inputs(dtype=jnp.float32):
    c = CONFIG
    ks = jax.random.split(jax.random.key(0), 7)
    state = jax.random.uniform(
        ks[0], (c['batch'], c['dim1'], c['dim2'], c['num_states']), dtype)
    w1 = jax.random.normal(ks[1], (c['num_states'], c['hidden1']), dtype) * 0.02
    b1 = jax.random.normal(ks[2], (c['hidden1'],), dtype) * 0.02
    w2 = jax.random.normal(ks[3], (c['hidden1'], c['hidden2']), dtype) * 0.02
    b2 = jax.random.normal(ks[4], (c['hidden2'],), dtype) * 0.02
    w3 = jax.random.normal(ks[5], (c['hidden2'], c['num_actions']), dtype) * 0.02
    b3 = jax.random.normal(ks[6], (c['num_actions'],), dtype) * 0.02
    return state, w1, b1, w2, b2, w3, b3


def workload(state, w1, b1, w2, b2, w3, b3):
    h = jax.nn.relu(state @ w1 + b1)
    h = jax.nn.relu(h @ w2 + b2)
    return h @ w3 + b3
