"""MLP_16942: flatten -> linear (bias) -> ReLU -> linear (bias)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'MLP_16942',
    'batch': 8192,
    'num_inputs': 2048,
    'width': 2048,
    'num_outputs': 2048,
    'depth': 1,
    'weight_scale': 0.02,
}


def create_inputs(dtype=jnp.float32):
    keys = jax.random.split(jax.random.key(0), 5)
    scale = CONFIG['weight_scale']
    x = jax.random.uniform(keys[0], (CONFIG['batch'], CONFIG['num_inputs']), dtype)
    w_first = jax.random.normal(keys[1], (CONFIG['num_inputs'], CONFIG['width']), dtype) * scale
    b_first = jax.random.normal(keys[2], (CONFIG['width'],), dtype) * scale
    w_last = jax.random.normal(keys[3], (CONFIG['width'], CONFIG['num_outputs']), dtype) * scale
    b_last = jax.random.normal(keys[4], (CONFIG['num_outputs'],), dtype) * scale
    return x, w_first, b_first, w_last, b_last


def workload(x, w_first, b_first, w_last, b_last):
    out = jnp.reshape(x, (x.shape[0], -1))
    out = out @ w_first + b_first
    out = jnp.maximum(out, 0)
    out = out @ w_last + b_last
    return out
