"""Critic_5653: Linear(input_size, 200) -> ReLU -> Linear(200, 1) over the last axis."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'Critic_5653',
    'batch': 8,
    'dim1': 128,
    'dim2': 128,
    'input_size': 512,
    'hidden': 200,
    'out_features': 1,
}


def create_inputs(dtype=jnp.float32):
    kx, k1, kb1, k2, kb2 = jax.random.split(jax.random.key(0), 5)
    c = CONFIG
    x = jax.random.uniform(kx, (c['batch'], c['dim1'], c['dim2'], c['input_size']), dtype)
    w1 = jax.random.normal(k1, (c['input_size'], c['hidden']), dtype) * 0.02
    b1 = jax.random.normal(kb1, (c['hidden'],), dtype) * 0.02
    w2 = jax.random.normal(k2, (c['hidden'], c['out_features']), dtype) * 0.02
    b2 = jax.random.normal(kb2, (c['out_features'],), dtype) * 0.02
    return x, w1, b1, w2, b2


def workload(x, w1, b1, w2, b2):
    h = jax.nn.relu(x @ w1 + b1)
    return h @ w2 + b2
