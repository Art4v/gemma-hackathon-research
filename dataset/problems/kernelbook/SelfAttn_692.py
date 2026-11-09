"""SelfAttn_692: Linear(H->1) query scores, softmax over time axis, weighted sum of values over time."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'SelfAttn_692',
    'batch': 8,
    'time': 256,
    'extra': 128,
    'hidden_size': 128,
}


def create_inputs(dtype=jnp.float32):
    k_keys, k_vals, k_w, k_b = jax.random.split(jax.random.key(0), 4)
    shape = (CONFIG['batch'], CONFIG['time'], CONFIG['extra'], CONFIG['hidden_size'])
    keys = jax.random.uniform(k_keys, shape, dtype)
    values = jax.random.uniform(k_vals, shape, dtype)
    w = jax.random.normal(k_w, (CONFIG['hidden_size'], 1), dtype) * 0.02
    b = jax.random.normal(k_b, (1,), dtype) * 0.02
    return keys, values, w, b


def workload(keys, values, w, b):
    scores = keys @ w + b  # (B, T, X, 1)
    alpha = jax.nn.softmax(scores, axis=1)
    return jnp.sum(values * alpha, axis=1)  # (B, X, H)
