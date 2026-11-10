"""Critic MLP: Linear -> ReLU -> concat(action) -> Linear -> ReLU -> Linear."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'Critic_3777',
    'batch': 16384,
    'state_size': 1024,
    'action_size': 128,
    'fcs1_units': 512,
    'fc2_units': 384,
    'out_units': 1,
}


def create_inputs(dtype=jnp.float32):
    c = CONFIG
    ks = jax.random.split(jax.random.key(0), 8)
    state = jax.random.uniform(ks[0], (c['batch'], c['state_size']), dtype)
    action = jax.random.uniform(ks[1], (c['batch'], c['action_size']), dtype)
    w1 = jax.random.normal(ks[2], (c['state_size'], c['fcs1_units']), dtype) * 0.02
    b1 = jax.random.normal(ks[3], (c['fcs1_units'],), dtype) * 0.02
    w2 = jax.random.normal(ks[4], (c['fcs1_units'] + c['action_size'], c['fc2_units']), dtype) * 0.02
    b2 = jax.random.normal(ks[5], (c['fc2_units'],), dtype) * 0.02
    w3 = jax.random.normal(ks[6], (c['fc2_units'], c['out_units']), dtype) * 0.02
    b3 = jax.random.normal(ks[7], (c['out_units'],), dtype) * 0.02
    return state, action, w1, b1, w2, b2, w3, b3


def workload(state, action, w1, b1, w2, b2, w3, b3):
    xs = jax.nn.relu(state @ w1 + b1)
    x = jnp.concatenate([xs, action], axis=1)
    x = jax.nn.relu(x @ w2 + b2)
    return x @ w3 + b3
