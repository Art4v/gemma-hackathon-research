"""PGNetwork: Linear(state_dim->hidden) -> ReLU -> Linear(hidden->action_dim) over the last axis."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'PGNetwork_6380',
    'input_shape': (16, 128, 128, 128),
    'state_dim': 128,
    'hidden_dim': 20,
    'action_dim': 128,
}


def create_inputs(dtype=jnp.float32):
    k_x, k_w1, k_b1, k_w2, k_b2 = jax.random.split(jax.random.key(0), 5)
    x = jax.random.uniform(k_x, CONFIG['input_shape'], dtype)
    w1 = jax.random.normal(k_w1, (CONFIG['state_dim'], CONFIG['hidden_dim']), dtype) * 0.02
    b1 = jax.random.normal(k_b1, (CONFIG['hidden_dim'],), dtype) * 0.02
    w2 = jax.random.normal(k_w2, (CONFIG['hidden_dim'], CONFIG['action_dim']), dtype) * 0.02
    b2 = jax.random.normal(k_b2, (CONFIG['action_dim'],), dtype) * 0.02
    return x, w1, b1, w2, b2


def workload(x, w1, b1, w2, b2):
    h = jax.nn.relu(x @ w1 + b1)
    return h @ w2 + b2
