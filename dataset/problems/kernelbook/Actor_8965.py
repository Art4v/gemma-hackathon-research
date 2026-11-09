"""Actor: concat -> Linear+ReLU -> Linear+ReLU -> Linear -> tanh*0.05*max_action + action -> clamp."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'Actor_8965',
    'batch': 16384,
    'state_dim': 2048,
    'action_dim': 1024,
    'hidden1': 400,
    'hidden2': 300,
    'max_action': 4.0,
    'residual_scale': 0.05,
}


def create_inputs(dtype=jnp.float32):
    c = CONFIG
    ks = jax.random.split(jax.random.key(0), 8)
    in_dim = c['state_dim'] + c['action_dim']
    state = jax.random.uniform(ks[0], (c['batch'], c['state_dim']), dtype)
    action = jax.random.uniform(ks[1], (c['batch'], c['action_dim']), dtype)
    w1 = jax.random.normal(ks[2], (in_dim, c['hidden1']), dtype) * 0.02
    b1 = jax.random.normal(ks[3], (c['hidden1'],), dtype) * 0.02
    w2 = jax.random.normal(ks[4], (c['hidden1'], c['hidden2']), dtype) * 0.02
    b2 = jax.random.normal(ks[5], (c['hidden2'],), dtype) * 0.02
    w3 = jax.random.normal(ks[6], (c['hidden2'], c['action_dim']), dtype) * 0.02
    b3 = jax.random.normal(ks[7], (c['action_dim'],), dtype) * 0.02
    return state, action, w1, b1, w2, b2, w3, b3


def workload(state, action, w1, b1, w2, b2, w3, b3):
    max_action = CONFIG['max_action']
    x = jnp.concatenate([state, action], axis=1)
    a = jax.nn.relu(x @ w1 + b1)
    a = jax.nn.relu(a @ w2 + b2)
    a = CONFIG['residual_scale'] * max_action * jnp.tanh(a @ w3 + b3)
    return jnp.clip(a + action, -max_action, max_action)
