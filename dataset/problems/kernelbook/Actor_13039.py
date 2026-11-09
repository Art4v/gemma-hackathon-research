"""Actor MLP: Linear -> ReLU -> Linear -> ReLU -> Linear -> Tanh."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'Actor_13039',
    'batch': 32768,
    'state_size': 1024,
    'fc_units': 1024,
    'fc_units2': 512,
    'action_size': 128,
}


def create_inputs(dtype=jnp.float32):
    keys = jax.random.split(jax.random.key(0), 7)
    b, s = CONFIG['batch'], CONFIG['state_size']
    h1, h2, a = CONFIG['fc_units'], CONFIG['fc_units2'], CONFIG['action_size']
    state = jax.random.uniform(keys[0], (b, s), dtype)
    w1 = jax.random.normal(keys[1], (s, h1), dtype) * 0.02
    b1 = jax.random.normal(keys[2], (h1,), dtype) * 0.02
    w2 = jax.random.normal(keys[3], (h1, h2), dtype) * 0.02
    b2 = jax.random.normal(keys[4], (h2,), dtype) * 0.02
    w3 = jax.random.normal(keys[5], (h2, a), dtype) * 0.02
    b3 = jax.random.normal(keys[6], (a,), dtype) * 0.02
    return state, w1, b1, w2, b2, w3, b3


def workload(state, w1, b1, w2, b2, w3, b3):
    x = jax.nn.relu(state @ w1 + b1)
    x = jax.nn.relu(x @ w2 + b2)
    return jnp.tanh(x @ w3 + b3)
