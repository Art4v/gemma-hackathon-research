"""Flatten -> Linear(m*n, 15) -> ReLU -> Linear(15, num_actions) (DQN_hot1 MLP)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'DQN_hot1_17174',
    'batch': 4096,
    'm': 128,
    'n': 128,
    'hidden': 15,  # hard-coded in the PyTorch module
    'num_actions': 128,
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    kx, k1, kb1, k2, kb2 = jax.random.split(key, 5)
    b, m, n = CONFIG['batch'], CONFIG['m'], CONFIG['n']
    h, a = CONFIG['hidden'], CONFIG['num_actions']
    x = jax.random.uniform(kx, (b, m, n), dtype)
    w1 = jax.random.normal(k1, (m * n, h), dtype) * 0.02
    b1 = jax.random.normal(kb1, (h,), dtype) * 0.02
    w2 = jax.random.normal(k2, (h, a), dtype) * 0.02
    b2 = jax.random.normal(kb2, (a,), dtype) * 0.02
    return x, w1, b1, w2, b2


def workload(x, w1, b1, w2, b2):
    x = x.reshape(x.shape[0], -1)
    h = jax.nn.relu(x @ w1 + b1)
    return h @ w2 + b2
