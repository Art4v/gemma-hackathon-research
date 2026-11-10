"""Two-layer MLP body with action injection: linear, relu, concat action, linear, relu."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'TwoLayerFCBodyWithAction_585',
    'batch': 8192,
    'state_dim': 2048,
    'action_dim': 1024,
    'hidden_size1': 2048,
    'hidden_size2': 2048,
    'weight_scale': 0.02,
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    k_x, k_action, k_w1, k_w2 = jax.random.split(key, 4)
    batch = CONFIG['batch']
    state_dim = CONFIG['state_dim']
    action_dim = CONFIG['action_dim']
    hidden_size1 = CONFIG['hidden_size1']
    hidden_size2 = CONFIG['hidden_size2']
    scale = CONFIG['weight_scale']
    x = jax.random.uniform(k_x, (batch, state_dim), dtype)
    action = jax.random.uniform(k_action, (batch, action_dim), dtype)
    w1 = jax.random.normal(k_w1, (state_dim, hidden_size1), dtype) * scale
    b1 = jnp.zeros((hidden_size1,), dtype)
    w2 = jax.random.normal(k_w2, (hidden_size1 + action_dim, hidden_size2), dtype) * scale
    b2 = jnp.zeros((hidden_size2,), dtype)
    return x, action, w1, b1, w2, b2


def workload(x, action, w1, b1, w2, b2):
    h = jax.nn.relu(x @ w1 + b1)
    combined = jnp.concatenate([h, action], axis=1)
    phi = jax.nn.relu(combined @ w2 + b2)
    return phi
