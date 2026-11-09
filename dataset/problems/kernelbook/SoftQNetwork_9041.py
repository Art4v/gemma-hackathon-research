"""[bf16 variant] SoftQNetwork: concat(state, action) -> Linear -> ReLU -> Linear -> ReLU -> Linear(1)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'SoftQNetwork_9041',
    'batch': 16384,
    'num_inputs': 2048,
    'num_actions': 2048,
    'hidden_size': 4096,
    'out_features': 1,
}


def create_inputs(dtype=jnp.bfloat16):
    b = CONFIG['batch']
    s = CONFIG['num_inputs']
    a = CONFIG['num_actions']
    h = CONFIG['hidden_size']
    o = CONFIG['out_features']
    keys = jax.random.split(jax.random.key(0), 8)
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
    x = jnp.concatenate([state, action], axis=1)
    x = jax.nn.relu(x @ w1 + b1)
    x = jax.nn.relu(x @ w2 + b2)
    return x @ w3 + b3
