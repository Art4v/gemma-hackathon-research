"""[bf16 variant] Linear -> softplus -> Linear -> softplus on the last axis (PointwiseConvolutionLayer)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'PointwiseConvolutionLayer_4360',
    'batch': 32,
    'dim1': 64,
    'dim2': 128,
    'F': 128,
    'hidden': 128,
    'F_prime': 128,
}


def create_inputs(dtype=jnp.bfloat16):
    c = CONFIG
    kx, k1, kb1, k2, kb2 = jax.random.split(jax.random.key(0), 5)
    x = jax.random.uniform(kx, (c['batch'], c['dim1'], c['dim2'], c['F']), dtype)
    w1 = jax.random.normal(k1, (c['F'], c['hidden']), dtype) * 0.02
    b1 = jax.random.normal(kb1, (c['hidden'],), dtype) * 0.02
    w2 = jax.random.normal(k2, (c['hidden'], c['F_prime']), dtype) * 0.02
    b2 = jax.random.normal(kb2, (c['F_prime'],), dtype) * 0.02
    return x, w1, b1, w2, b2


def workload(x, w1, b1, w2, b2):
    h = jax.nn.softplus(x @ w1 + b1)
    return jax.nn.softplus(h @ w2 + b2)
