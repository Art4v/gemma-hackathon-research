"""DQN_hot4: flatten -> Linear -> ReLU -> Linear -> ReLU -> Linear."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'DQN_hot4_17173',
    'batch': 16384,
    'm': 64,
    'n': 64,
    'hidden1': 100,
    'hidden2': 25,
    'num_actions': 4,
}


def create_inputs(dtype=jnp.float32):
    keys = jax.random.split(jax.random.key(0), 7)
    b, m, n = CONFIG['batch'], CONFIG['m'], CONFIG['n']
    h1, h2, a = CONFIG['hidden1'], CONFIG['hidden2'], CONFIG['num_actions']
    d = m * n
    x = jax.random.uniform(keys[0], (b, m, n), dtype)
    w1 = jax.random.normal(keys[1], (d, h1), dtype) * 0.02
    b1 = jax.random.normal(keys[2], (h1,), dtype) * 0.02
    w2 = jax.random.normal(keys[3], (h1, h2), dtype) * 0.02
    b2 = jax.random.normal(keys[4], (h2,), dtype) * 0.02
    w3 = jax.random.normal(keys[5], (h2, a), dtype) * 0.02
    b3 = jax.random.normal(keys[6], (a,), dtype) * 0.02
    return x, w1, b1, w2, b2, w3, b3


def workload(x, w1, b1, w2, b2, w3, b3):
    x = x.reshape(x.shape[0], -1)
    x = jax.nn.relu(x @ w1 + b1)
    x = jax.nn.relu(x @ w2 + b2)
    return x @ w3 + b3
