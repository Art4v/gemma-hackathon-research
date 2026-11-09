"""TwoLayerNet: Linear -> ReLU (clamp min=0) -> Linear over the last axis."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'TwoLayerNet_15183',
    'batch': 8,
    'dim1': 64,
    'dim2': 256,
    'd_in': 512,
    'hidden': 1024,
    'd_out': 512,
}


def create_inputs(dtype=jnp.float32):
    c = CONFIG
    kx, k1, kb1, k2, kb2 = jax.random.split(jax.random.key(0), 5)
    x = jax.random.uniform(kx, (c['batch'], c['dim1'], c['dim2'], c['d_in']), dtype)
    w1 = jax.random.normal(k1, (c['d_in'], c['hidden']), dtype) * 0.02
    b1 = jax.random.normal(kb1, (c['hidden'],), dtype) * 0.02
    w2 = jax.random.normal(k2, (c['hidden'], c['d_out']), dtype) * 0.02
    b2 = jax.random.normal(kb2, (c['d_out'],), dtype) * 0.02
    return x, w1, b1, w2, b2


def workload(x, w1, b1, w2, b2):
    h = jnp.maximum(x @ w1 + b1, 0)
    return h @ w2 + b2
