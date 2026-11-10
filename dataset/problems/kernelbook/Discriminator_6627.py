"""[bf16 variant] Linear -> ReLU -> Linear(out_dim -> 1) -> Sigmoid (Discriminator)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'Discriminator_6627',
    'batch': 8,
    'd1': 32,
    'd2': 256,
    'in_dim': 1024,
    'out_dim': 1024,
}


def create_inputs(dtype=jnp.bfloat16):
    c = CONFIG
    kx, k1, kb1, k2, kb2 = jax.random.split(jax.random.key(0), 5)
    x = jax.random.uniform(kx, (c['batch'], c['d1'], c['d2'], c['in_dim']), dtype)
    w1 = jax.random.normal(k1, (c['in_dim'], c['out_dim']), dtype) * 0.02
    b1 = jax.random.normal(kb1, (c['out_dim'],), dtype) * 0.02
    w2 = jax.random.normal(k2, (c['out_dim'], 1), dtype) * 0.02
    b2 = jax.random.normal(kb2, (1,), dtype) * 0.02
    return x, w1, b1, w2, b2


def workload(x, w1, b1, w2, b2):
    h = jax.nn.relu(x @ w1 + b1)
    return jax.nn.sigmoid(h @ w2 + b2)
