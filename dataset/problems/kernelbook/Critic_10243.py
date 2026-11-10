"""Critic MLP: concat(state, action) -> Linear -> ReLU -> Linear -> ReLU -> Linear(1)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'Critic_10243',
    'batch': 32768,
    'state_size': 1024,
    'action_size': 512,
    'hidden_size': 1024,
    'out_size': 1,
}


def create_inputs(dtype=jnp.float32):
    keys = jax.random.split(jax.random.key(0), 8)
    b = CONFIG['batch']
    s = CONFIG['state_size']
    a = CONFIG['action_size']
    h = CONFIG['hidden_size']
    o = CONFIG['out_size']
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
    x = jnp.concatenate((state, action), axis=1)
    x = jax.nn.relu(x @ w1 + b1)
    x = jax.nn.relu(x @ w2 + b2)
    return x @ w3 + b3
