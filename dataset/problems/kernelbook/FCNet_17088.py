"""FCNet: weight-normalized (dim=None) Linear on the last axis, no activation, dropout identity."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'FCNet_17088',
    'batch': 8,
    'd1': 64,
    'd2': 128,
    'in_size': 1024,
    'out_size': 1024,
}


def create_inputs(dtype=jnp.float32):
    kx, kv, kb = jax.random.split(jax.random.key(0), 3)
    x = jax.random.uniform(
        kx, (CONFIG['batch'], CONFIG['d1'], CONFIG['d2'], CONFIG['in_size']), dtype)
    v = jax.random.normal(kv, (CONFIG['in_size'], CONFIG['out_size']), dtype) * 0.02
    # weight_norm init sets g = ||v||_F (scalar when dim=None)
    g = jnp.linalg.norm(v).astype(dtype)
    b = jax.random.normal(kb, (CONFIG['out_size'],), dtype) * 0.02
    return x, v, g, b


def workload(x, v, g, b):
    w = v * (g / jnp.sqrt(jnp.sum(v * v)))
    return jnp.matmul(x, w) + b
