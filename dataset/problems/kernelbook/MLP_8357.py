"""MLP: Linear -> ReLU -> Linear -> ReLU -> Linear on the last axis of a 4-D input."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'MLP_8357',
    'batch': 16,
    'dim1': 64,
    'dim2': 64,
    'input_dim': 512,
    'hidden_dim': 1024,
    'output_dim': 512,
}


def create_inputs(dtype=jnp.float32):
    c = CONFIG
    k = jax.random.split(jax.random.key(0), 7)
    x = jax.random.uniform(k[0], (c['batch'], c['dim1'], c['dim2'], c['input_dim']), dtype)
    w1 = jax.random.normal(k[1], (c['input_dim'], c['hidden_dim']), dtype) * 0.02
    b1 = jax.random.normal(k[2], (c['hidden_dim'],), dtype) * 0.02
    w2 = jax.random.normal(k[3], (c['hidden_dim'], c['hidden_dim']), dtype) * 0.02
    b2 = jax.random.normal(k[4], (c['hidden_dim'],), dtype) * 0.02
    w3 = jax.random.normal(k[5], (c['hidden_dim'], c['output_dim']), dtype) * 0.02
    b3 = jax.random.normal(k[6], (c['output_dim'],), dtype) * 0.02
    return x, w1, b1, w2, b2, w3, b3


def workload(x, w1, b1, w2, b2, w3, b3):
    h = jax.nn.relu(x @ w1 + b1)
    h = jax.nn.relu(h @ w2 + b2)
    return h @ w3 + b3
