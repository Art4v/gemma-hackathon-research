"""Linear -> ReLU -> Linear (out_features=1) MLP over the last axis (Generator_4837)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'Generator_4837',
    'batch': 16,
    'd1': 64,
    'd2': 256,
    'nz': 128,
    'hidden': 128,
    'out_features': 1,
}


def create_inputs(dtype=jnp.float32):
    c = CONFIG
    kx, k1, kb1, k2, kb2 = jax.random.split(jax.random.key(0), 5)
    x = jax.random.uniform(kx, (c['batch'], c['d1'], c['d2'], c['nz']), dtype)
    w1 = jax.random.normal(k1, (c['nz'], c['hidden']), dtype) * 0.02
    b1 = jax.random.normal(kb1, (c['hidden'],), dtype) * 0.02
    w2 = jax.random.normal(k2, (c['hidden'], c['out_features']), dtype) * 0.02
    b2 = jax.random.normal(kb2, (c['out_features'],), dtype) * 0.02
    return x, w1, b1, w2, b2


def workload(x, w1, b1, w2, b2):
    h = jax.nn.relu(x @ w1 + b1)
    return h @ w2 + b2
