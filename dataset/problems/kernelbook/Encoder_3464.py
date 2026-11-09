"""Encoder: concat(x, c) -> Linear -> ReLU -> Dropout(identity) -> Linear -> ReLU."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'Encoder_3464',
    'batch': 8192,
    'sample_size': 2048,
    'condition_size': 2048,
    'hidden_size': 4096,
}


def create_inputs(dtype=jnp.float32):
    b = CONFIG['batch']
    s = CONFIG['sample_size']
    c = CONFIG['condition_size']
    h = CONFIG['hidden_size']
    kx, kc, kw1, kb1, kw3, kb3 = jax.random.split(jax.random.key(0), 6)
    x = jax.random.uniform(kx, (b, s), dtype)
    cond = jax.random.uniform(kc, (b, c), dtype)
    w1 = jax.random.normal(kw1, (s + c, h), dtype) * 0.02
    b1 = jax.random.normal(kb1, (h,), dtype) * 0.02
    w3 = jax.random.normal(kw3, (h, h), dtype) * 0.02
    b3 = jax.random.normal(kb3, (h,), dtype) * 0.02
    return x, cond, w1, b1, w3, b3


def workload(x, c, w1, b1, w3, b3):
    h = jnp.concatenate([x, c], axis=1)
    h = jax.nn.relu(h @ w1 + b1)
    # Dropout is the identity in inference mode.
    return jax.nn.relu(h @ w3 + b3)
