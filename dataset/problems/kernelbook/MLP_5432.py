"""[bf16 variant] MLP: Linear -> ReLU -> Linear -> ReLU -> Linear -> Sigmoid over the last axis."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'MLP_5432',
    'batch': 32,
    'd1': 128,
    'd2': 128,
    'state_dim': 128,
    'hidden': 36,
    'out_dim': 1,
}


def create_inputs(dtype=jnp.bfloat16):
    keys = jax.random.split(jax.random.key(0), 7)
    c = CONFIG
    x = jax.random.uniform(keys[0], (c['batch'], c['d1'], c['d2'], c['state_dim']), dtype)
    w1 = jax.random.normal(keys[1], (c['state_dim'], c['hidden']), dtype) * 0.02
    b1 = jax.random.normal(keys[2], (c['hidden'],), dtype) * 0.02
    w2 = jax.random.normal(keys[3], (c['hidden'], c['hidden']), dtype) * 0.02
    b2 = jax.random.normal(keys[4], (c['hidden'],), dtype) * 0.02
    w3 = jax.random.normal(keys[5], (c['hidden'], c['out_dim']), dtype) * 0.02
    b3 = jax.random.normal(keys[6], (c['out_dim'],), dtype) * 0.02
    return x, w1, b1, w2, b2, w3, b3


def workload(x, w1, b1, w2, b2, w3, b3):
    h = jax.nn.relu(x @ w1 + b1)
    h = jax.nn.relu(h @ w2 + b2)
    return jax.nn.sigmoid(h @ w3 + b3)
