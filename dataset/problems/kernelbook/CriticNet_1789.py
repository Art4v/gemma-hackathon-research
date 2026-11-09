"""CriticNet: Linear -> ReLU -> Linear -> ReLU -> Linear(->1) -> squeeze."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'CriticNet_1789',
    'batch_dims': (32, 128, 128),
    'state_size': 128,
    'fc1_units': 128,
    'fc2_units': 128,
    'out_units': 1,
}


def create_inputs(dtype=jnp.float32):
    ks = jax.random.split(jax.random.key(0), 7)
    s, h1, h2, o = CONFIG['state_size'], CONFIG['fc1_units'], CONFIG['fc2_units'], CONFIG['out_units']
    state = jax.random.uniform(ks[0], (*CONFIG['batch_dims'], s), dtype)
    w1 = jax.random.normal(ks[1], (s, h1), dtype) * 0.02
    b1 = jax.random.normal(ks[2], (h1,), dtype) * 0.02
    w2 = jax.random.normal(ks[3], (h1, h2), dtype) * 0.02
    b2 = jax.random.normal(ks[4], (h2,), dtype) * 0.02
    w3 = jax.random.normal(ks[5], (h2, o), dtype) * 0.02
    b3 = jax.random.normal(ks[6], (o,), dtype) * 0.02
    return state, w1, b1, w2, b2, w3, b3


def workload(state, w1, b1, w2, b2, w3, b3):
    x = jax.nn.relu(state @ w1 + b1)
    x = jax.nn.relu(x @ w2 + b2)
    v = x @ w3 + b3
    return jnp.squeeze(v)
