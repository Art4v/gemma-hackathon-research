"""ValueNetwork: Linear -> ReLU -> Linear -> ReLU -> Linear(hidden -> 1) over the last axis."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'ValueNetwork_338',
    'batch': 8,
    'dim1': 128,
    'dim2': 128,
    'num_inputs': 512,
    'hidden_dim': 512,
    'out_dim': 1,
}


def create_inputs(dtype=jnp.float32):
    keys = jax.random.split(jax.random.key(0), 7)
    c = CONFIG
    state = jax.random.uniform(
        keys[0], (c['batch'], c['dim1'], c['dim2'], c['num_inputs']), dtype)
    w1 = jax.random.normal(keys[1], (c['num_inputs'], c['hidden_dim']), dtype) * 0.02
    b1 = jax.random.normal(keys[2], (c['hidden_dim'],), dtype) * 0.02
    w2 = jax.random.normal(keys[3], (c['hidden_dim'], c['hidden_dim']), dtype) * 0.02
    b2 = jax.random.normal(keys[4], (c['hidden_dim'],), dtype) * 0.02
    w3 = jax.random.normal(keys[5], (c['hidden_dim'], c['out_dim']), dtype) * 0.02
    b3 = jax.random.normal(keys[6], (c['out_dim'],), dtype) * 0.02
    return state, w1, b1, w2, b2, w3, b3


def workload(state, w1, b1, w2, b2, w3, b3):
    x = jax.nn.relu(state @ w1 + b1)
    x = jax.nn.relu(x @ w2 + b2)
    return x @ w3 + b3
