"""Critic: concat(state, action) -> Linear -> ReLU -> Linear -> ReLU -> Linear(1)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'Critic_13001',
    'batch': 65536,
    'state_dim': 512,
    'action_dim': 512,
    'hidden': 128,
    'out_dim': 1,
}


def create_inputs(dtype=jnp.float32):
    keys = jax.random.split(jax.random.key(0), 8)
    b = CONFIG['batch']
    s, a = CONFIG['state_dim'], CONFIG['action_dim']
    h, o = CONFIG['hidden'], CONFIG['out_dim']
    state = jax.random.uniform(keys[0], (b, s), dtype)
    action = jax.random.uniform(keys[1], (b, a), dtype)
    w1 = jax.random.normal(keys[2], (s + a, h), dtype) * 0.02
    b1 = jax.random.normal(keys[3], (h,), dtype) * 0.02
    w2 = jax.random.normal(keys[4], (h, h), dtype) * 0.02
    b2 = jax.random.normal(keys[5], (h,), dtype) * 0.02
    w3 = jax.random.normal(keys[6], (h, o), dtype) * 0.02
    b3 = jax.random.normal(keys[7], (o,), dtype) * 0.02
    return state, action, w1, b1, w2, b2, w3, b3


def workload(state, action, w1, b1, w2, b2, w3, b3):
    sa = jnp.concatenate([state, action], axis=1)
    q = jax.nn.relu(sa @ w1 + b1)
    q = jax.nn.relu(q @ w2 + b2)
    return q @ w3 + b3
