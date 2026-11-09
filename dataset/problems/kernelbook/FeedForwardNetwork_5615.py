"""Linear -> tanh-GELU -> Dropout (identity) -> Linear feed-forward network."""
import math

import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'FeedForwardNetwork_5615',
    'batch': 8,
    'seq_len': 2048,
    'in_dim': 1024,
    'hid_dim': 4096,
    'gelu_coeff': 0.044715,
}


def create_inputs(dtype=jnp.float32):
    k_x, k_w1, k_b1, k_w2, k_b2 = jax.random.split(jax.random.key(0), 5)
    b, s = CONFIG['batch'], CONFIG['seq_len']
    d, h = CONFIG['in_dim'], CONFIG['hid_dim']
    x = jax.random.uniform(k_x, (b, s, d), dtype)
    w1 = jax.random.normal(k_w1, (d, h), dtype) * 0.02
    b1 = jax.random.normal(k_b1, (h,), dtype) * 0.02
    w2 = jax.random.normal(k_w2, (h, d), dtype) * 0.02
    b2 = jax.random.normal(k_b2, (d,), dtype) * 0.02
    return x, w1, b1, w2, b2


def workload(x, w1, b1, w2, b2):
    hid = x @ w1 + b1
    c = math.sqrt(2.0 / math.pi)
    hid = 0.5 * hid * (1.0 + jnp.tanh(c * (hid + CONFIG['gelu_coeff'] * hid ** 3)))
    # Dropout is identity in inference mode.
    return hid @ w2 + b2
