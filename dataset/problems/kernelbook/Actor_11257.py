"""[bf16 variant] Actor MLP: Linear -> ReLU -> Linear -> ReLU -> Linear -> Tanh over the last dim."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'Actor_11257',
    'lead_shape': (16, 32, 64),
    'state_size': 512,
    'fc1_units': 550,
    'fc2_units': 300,
    'action_size': 128,
}


def create_inputs(dtype=jnp.bfloat16):
    keys = jax.random.split(jax.random.key(0), 7)
    s, h1, h2, a = (CONFIG['state_size'], CONFIG['fc1_units'],
                    CONFIG['fc2_units'], CONFIG['action_size'])
    state = jax.random.uniform(keys[0], (*CONFIG['lead_shape'], s), dtype)
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
