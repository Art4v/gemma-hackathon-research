"""Actor MLP: Linear -> ReLU -> Linear -> ReLU -> Linear -> Sigmoid."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'Actor_17916',
    'input_shape': (16, 32, 128, 512),
    'nb_states': 512,
    'hidden1': 512,
    'hidden2': 384,
    'nb_actions': 128,
}


def create_inputs(dtype=jnp.float32):
    keys = jax.random.split(jax.random.key(0), 7)
    s, h1, h2, a = CONFIG['nb_states'], CONFIG['hidden1'], CONFIG['hidden2'], CONFIG['nb_actions']
    x = jax.random.uniform(keys[0], CONFIG['input_shape'], dtype)
    w1 = jax.random.normal(keys[1], (s, h1), dtype) * 0.02
    b1 = jax.random.normal(keys[2], (h1,), dtype) * 0.02
    w2 = jax.random.normal(keys[3], (h1, h2), dtype) * 0.02
    b2 = jax.random.normal(keys[4], (h2,), dtype) * 0.02
    w3 = jax.random.normal(keys[5], (h2, a), dtype) * 0.02
    b3 = jax.random.normal(keys[6], (a,), dtype) * 0.02
    return x, w1, b1, w2, b2, w3, b3


def workload(x, w1, b1, w2, b2, w3, b3):
    out = jax.nn.relu(x @ w1 + b1)
    out = jax.nn.relu(out @ w2 + b2)
    return jax.nn.sigmoid(out @ w3 + b3)
