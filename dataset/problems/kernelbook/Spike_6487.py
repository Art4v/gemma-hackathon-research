"""[bf16 variant] Spike activation: alpha*x + beta*(clamped triangle - 2*shifted clamped triangle)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'Spike_6487',
    'batch': 16,
    'rows': 4096,
    'cols': 1024,
    'center': 1.0,
    'width': 1.0,
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    x = jax.random.uniform(key, (CONFIG['batch'], CONFIG['rows'], CONFIG['cols']), dtype)
    alpha = jnp.ones((1,), dtype)
    beta = jnp.ones((1,), dtype)
    return x, alpha, beta


def workload(x, alpha, beta):
    c = CONFIG['center']
    w = CONFIG['width']
    zero = jnp.zeros_like(x)
    tri = jnp.minimum(jnp.maximum(x - (c - w), zero), jnp.maximum(-x + (c + w), zero))
    tri_shift = jnp.minimum(jnp.maximum(x - (c - w + 1), zero), jnp.maximum(-x + (c + w + 1), zero))
    return alpha * x + beta * (tri - 2 * tri_shift)
