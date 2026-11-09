"""Sys_R: concat(state, next_state, action) -> Linear -> ReLU -> Linear -> ReLU -> Linear(1)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'Sys_R_2630',
    'batch': 8192,
    'state_size': 1024,
    'action_size': 1024,
    'fc1_units': 256,
    'fc2_units': 256,
    'out_units': 1,
}


def create_inputs(dtype=jnp.float32):
    c = CONFIG
    b, s, a = c['batch'], c['state_size'], c['action_size']
    in_dim = 2 * s + a
    ks = jax.random.split(jax.random.key(0), 9)
    state = jax.random.uniform(ks[0], (b, s), dtype)
    next_state = jax.random.uniform(ks[1], (b, s), dtype)
    action = jax.random.uniform(ks[2], (b, a), dtype)
    w1 = jax.random.normal(ks[3], (in_dim, c['fc1_units']), dtype) * 0.02
    b1 = jax.random.normal(ks[4], (c['fc1_units'],), dtype) * 0.02
    w2 = jax.random.normal(ks[5], (c['fc1_units'], c['fc2_units']), dtype) * 0.02
    b2 = jax.random.normal(ks[6], (c['fc2_units'],), dtype) * 0.02
    w3 = jax.random.normal(ks[7], (c['fc2_units'], c['out_units']), dtype) * 0.02
    b3 = jax.random.normal(ks[8], (c['out_units'],), dtype) * 0.02
    return state, next_state, action, w1, b1, w2, b2, w3, b3


def workload(state, next_state, action, w1, b1, w2, b2, w3, b3):
    xa = jnp.concatenate([state, next_state, action], axis=1)
    x = jax.nn.relu(xa @ w1 + b1)
    x = jax.nn.relu(x @ w2 + b2)
    return x @ w3 + b3
