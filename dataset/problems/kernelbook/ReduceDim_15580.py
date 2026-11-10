"""ReduceDim: Linear (x @ w + b) followed by L2 normalize along the last dim."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'ReduceDim_15580',
    'batch': 16,
    'height': 64,
    'width': 64,
    'input_dimension': 1024,
    'output_dimension': 1024,
    'eps': 1e-12,
}


def create_inputs(dtype=jnp.float32):
    kx, kw, kb = jax.random.split(jax.random.key(0), 3)
    c = CONFIG
    x = jax.random.uniform(
        kx, (c['batch'], c['height'], c['width'], c['input_dimension']), dtype)
    w = jax.random.normal(kw, (c['input_dimension'], c['output_dimension']), dtype) * 0.02
    b = jax.random.normal(kb, (c['output_dimension'],), dtype) * 0.02
    return x, w, b


def workload(x, w, b):
    y = x @ w + b
    norm = jnp.sqrt(jnp.sum(y * y, axis=-1, keepdims=True))
    return y / jnp.maximum(norm, CONFIG['eps'])
