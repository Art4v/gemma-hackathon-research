"""Linear -> ReLU -> concat(action) -> Linear -> ReLU (TwoLayerFCBodyWithAction)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'TwoLayerFCBodyWithAction_9096',
    'batch': 16384,
    'state_dim': 2048,
    'action_dim': 1024,
    'hidden1': 2048,
    'hidden2': 2048,
}


def create_inputs(dtype=jnp.float32):
    kx, ka, k1, k2 = jax.random.split(jax.random.key(0), 4)
    b = CONFIG['batch']
    s, a = CONFIG['state_dim'], CONFIG['action_dim']
    h1, h2 = CONFIG['hidden1'], CONFIG['hidden2']
    x = jax.random.uniform(kx, (b, s), dtype)
    action = jax.random.uniform(ka, (b, a), dtype)
    w1 = jax.random.normal(k1, (s, h1), dtype) * 0.02
    b1 = jnp.zeros((h1,), dtype)
    w2 = jax.random.normal(k2, (h1 + a, h2), dtype) * 0.02
    b2 = jnp.zeros((h2,), dtype)
    return x, action, w1, b1, w2, b2


def workload(x, action, w1, b1, w2, b2):
    h = jax.nn.relu(x @ w1 + b1)
    return jax.nn.relu(jnp.concatenate([h, action], axis=1) @ w2 + b2)
