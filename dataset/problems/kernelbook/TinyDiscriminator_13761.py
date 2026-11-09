"""[bf16 variant] TinyDiscriminator: Linear -> LeakyReLU(0.1) -> Linear(d_hidden, 1) over the last axis."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'TinyDiscriminator_13761',
    'batch': 16,
    'dim1': 128,
    'dim2': 128,
    'n_features': 256,
    'd_hidden': 128,
    'n_out': 1,
    'negative_slope': 0.1,
}


def create_inputs(dtype=jnp.bfloat16):
    c = CONFIG
    kx, k1, kb1, k2, kb2 = jax.random.split(jax.random.key(0), 5)
    x = jax.random.uniform(kx, (c['batch'], c['dim1'], c['dim2'], c['n_features']), dtype)
    w1 = jax.random.normal(k1, (c['n_features'], c['d_hidden']), dtype) * 0.02
    b1 = jax.random.normal(kb1, (c['d_hidden'],), dtype) * 0.02
    w2 = jax.random.normal(k2, (c['d_hidden'], c['n_out']), dtype) * 0.02
    b2 = jax.random.normal(kb2, (c['n_out'],), dtype) * 0.02
    return x, w1, b1, w2, b2


def workload(x, w1, b1, w2, b2):
    h = x @ w1 + b1
    features = jnp.where(h >= 0, h, CONFIG['negative_slope'] * h)
    return features @ w2 + b2
