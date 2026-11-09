"""[bf16 variant] Actor MLP: Linear -> ReLU -> Linear -> ReLU -> Linear -> Tanh over the last axis."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'Actor_9677',
    'lead_shape': (16, 64, 128),
    'state_size': 128,
    'fc_units': 256,
    'fc_units2': 128,
    'action_size': 128,
}


def create_inputs(dtype=jnp.bfloat16):
    k = jax.random.split(jax.random.key(0), 7)
    s, h1, h2, a = (CONFIG['state_size'], CONFIG['fc_units'],
                    CONFIG['fc_units2'], CONFIG['action_size'])
    state = jax.random.uniform(k[0], (*CONFIG['lead_shape'], s), dtype)
    w1 = jax.random.normal(k[1], (s, h1), dtype) * 0.02
    b1 = jax.random.normal(k[2], (h1,), dtype) * 0.02
    w2 = jax.random.normal(k[3], (h1, h2), dtype) * 0.02
    b2 = jax.random.normal(k[4], (h2,), dtype) * 0.02
    w3 = jax.random.normal(k[5], (h2, a), dtype) * 0.02
    b3 = jax.random.normal(k[6], (a,), dtype) * 0.02
    return state, w1, b1, w2, b2, w3, b3


def workload(state, w1, b1, w2, b2, w3, b3):
    x = jax.nn.relu(state @ w1 + b1)
    x = jax.nn.relu(x @ w2 + b2)
    return jnp.tanh(x @ w3 + b3)
