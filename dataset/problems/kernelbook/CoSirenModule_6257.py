"""CoSiren: Linear(in -> out/2) -> concat(sin, cos) - pi/4."""
import math

import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'CoSirenModule_6257',
    'batch': 8,
    'd1': 128,
    'd2': 128,
    'in_features': 512,
    'out_features': 1024,
    'offset': math.pi / 4,
}


def create_inputs(dtype=jnp.float32):
    kx, kw, kb = jax.random.split(jax.random.key(0), 3)
    half = CONFIG['out_features'] // 2
    x = jax.random.uniform(
        kx, (CONFIG['batch'], CONFIG['d1'], CONFIG['d2'], CONFIG['in_features']), dtype)
    w = jax.random.normal(kw, (CONFIG['in_features'], half), dtype) * 0.02
    b = jax.random.normal(kb, (half,), dtype) * 0.02
    return x, w, b


def workload(x, w, b):
    h = x @ w + b
    return jnp.concatenate([jnp.sin(h), jnp.cos(h)], axis=-1) - CONFIG['offset']
