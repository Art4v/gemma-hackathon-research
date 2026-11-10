"""Maxout: elementwise max of two Linear layers (x @ w1 + b1, x @ w2 + b2)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'Maxout_4302',
    'batch': 8,
    'dim1': 64,
    'dim2': 64,
    'in_features': 1024,
    'out_features': 1024,
}


def create_inputs(dtype=jnp.float32):
    c = CONFIG
    k = jax.random.split(jax.random.key(0), 5)
    x = jax.random.uniform(k[0], (c['batch'], c['dim1'], c['dim2'], c['in_features']), dtype)
    w1 = jax.random.normal(k[1], (c['in_features'], c['out_features']), dtype) * 0.02
    b1 = jax.random.normal(k[2], (c['out_features'],), dtype) * 0.02
    w2 = jax.random.normal(k[3], (c['in_features'], c['out_features']), dtype) * 0.02
    b2 = jax.random.normal(k[4], (c['out_features'],), dtype) * 0.02
    return x, w1, b1, w2, b2


def workload(x, w1, b1, w2, b2):
    return jnp.maximum(x @ w1 + b1, x @ w2 + b2)
