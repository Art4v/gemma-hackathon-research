"""Flatten -> Linear -> ReLU -> Linear (UnderfitNet two-layer MLP)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'UnderfitNet_17600',
    'batch': 16384,
    'in_features': 1024,
    'hidden_features': 2048,
    'out_features': 1024,
    'weight_scale': 0.02,
}


def create_inputs(dtype=jnp.float32):
    kx, kw1, kb1, kw2, kb2 = jax.random.split(jax.random.key(0), 5)
    n = CONFIG['batch']
    d_in = CONFIG['in_features']
    d_hidden = CONFIG['hidden_features']
    d_out = CONFIG['out_features']
    scale = CONFIG['weight_scale']
    x = jax.random.uniform(kx, (n, d_in), dtype)
    w1 = jax.random.normal(kw1, (d_in, d_hidden), dtype) * scale
    b1 = jax.random.normal(kb1, (d_hidden,), dtype) * scale
    w2 = jax.random.normal(kw2, (d_hidden, d_out), dtype) * scale
    b2 = jax.random.normal(kb2, (d_out,), dtype) * scale
    return x, w1, b1, w2, b2


def workload(x, w1, b1, w2, b2):
    x = jnp.reshape(x, (-1, w1.shape[0]))
    h = jax.nn.relu(x @ w1 + b1)
    return h @ w2 + b2
