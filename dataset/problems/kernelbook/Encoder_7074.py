"""Encoder_7074: Linear -> ReLU -> Linear -> ReLU -> Linear -> view(-1, 2, M) -> (mu, exp(log_var / 2))."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'Encoder_7074',
    'x_shape': (8, 64, 64, 1024),
    'D': 1024,
    'H': 1024,
    'M': 512,
}


def create_inputs(dtype=jnp.float32):
    k = jax.random.split(jax.random.key(0), 4)
    D, H, M = CONFIG['D'], CONFIG['H'], CONFIG['M']
    x = jax.random.uniform(k[0], CONFIG['x_shape'], dtype)
    w1 = jax.random.normal(k[1], (D, H), dtype) * 0.02
    b1 = jnp.zeros((H,), dtype)
    w2 = jax.random.normal(k[2], (H, H), dtype) * 0.02
    b2 = jnp.zeros((H,), dtype)
    w3 = jax.random.normal(k[3], (H, 2 * M), dtype) * 0.02
    b3 = jnp.zeros((2 * M,), dtype)
    return x, w1, b1, w2, b2, w3, b3


def workload(x, w1, b1, w2, b2, w3, b3):
    # Linear acts on the last axis only; leading axes are folded by the view below.
    m = w3.shape[1] // 2
    h = jax.nn.relu(x @ w1 + b1)
    h = jax.nn.relu(h @ w2 + b2)
    y = (h @ w3 + b3).reshape(-1, 2, m)
    mu = y[:, 0, :]
    std = jnp.exp(y[:, 1, :] / 2)
    return mu, std
