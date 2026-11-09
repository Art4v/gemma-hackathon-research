"""Decoder2: Linear -> ReLU -> Linear -> ReLU -> Linear -> tanh, plus std = exp(log_scale)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'Decoder2_7068',
    'batch': 8,
    'dim1': 32,
    'dim2': 128,
    'M': 512,
    'H': 1024,
    'D': 512,
}


def create_inputs(dtype=jnp.float32):
    c = CONFIG
    keys = jax.random.split(jax.random.key(0), 7)
    z = jax.random.uniform(keys[0], (c['batch'], c['dim1'], c['dim2'], c['M']), dtype)
    w1 = jax.random.normal(keys[1], (c['M'], 2 * c['H']), dtype) * 0.02
    b1 = jax.random.normal(keys[2], (2 * c['H'],), dtype) * 0.02
    w2 = jax.random.normal(keys[3], (2 * c['H'], c['H']), dtype) * 0.02
    b2 = jax.random.normal(keys[4], (c['H'],), dtype) * 0.02
    w3 = jax.random.normal(keys[5], (c['H'], c['D']), dtype) * 0.02
    b3 = jax.random.normal(keys[6], (c['D'],), dtype) * 0.02
    log_scale = jnp.zeros((1,), dtype)
    return z, w1, b1, w2, b2, w3, b3, log_scale


def workload(z, w1, b1, w2, b2, w3, b3, log_scale):
    h = jax.nn.relu(z @ w1 + b1)
    h = jax.nn.relu(h @ w2 + b2)
    mu = jnp.tanh(h @ w3 + b3)
    std = jnp.exp(log_scale)
    return mu, std
