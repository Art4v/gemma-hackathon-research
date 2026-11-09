"""[bf16 variant] QNetworkSmall: Linear -> ReLU -> Linear -> ReLU -> Linear over the last axis."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'QNetworkSmall_14690',
    'lead_shape': (32, 64, 64),
    'state_size': 128,
    'hidden_size': 64,
    'action_size': 128,
}


def create_inputs(dtype=jnp.bfloat16):
    keys = jax.random.split(jax.random.key(0), 7)
    s, h, a = CONFIG['state_size'], CONFIG['hidden_size'], CONFIG['action_size']
    x = jax.random.uniform(keys[0], (*CONFIG['lead_shape'], s), dtype)
    w1 = jax.random.normal(keys[1], (s, h), dtype) * 0.02
    b1 = jax.random.normal(keys[2], (h,), dtype) * 0.02
    w2 = jax.random.normal(keys[3], (h, h), dtype) * 0.02
    b2 = jax.random.normal(keys[4], (h,), dtype) * 0.02
    w3 = jax.random.normal(keys[5], (h, a), dtype) * 0.02
    b3 = jax.random.normal(keys[6], (a,), dtype) * 0.02
    return x, w1, b1, w2, b2, w3, b3


def workload(x, w1, b1, w2, b2, w3, b3):
    h = jax.nn.relu(x @ w1 + b1)
    h = jax.nn.relu(h @ w2 + b2)
    return h @ w3 + b3
