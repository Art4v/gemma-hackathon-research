"""Normalize_4644: per-feature standardization over the last axis (mean, unbiased var, sqrt) with gain and bias."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'Normalize_4644',
    'batch': 8,
    'height': 128,
    'width': 256,
    'features': 256,
    'epsilon': 1e-06,
    'axis': -1,
    'ddof': 1,
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    shape = (CONFIG['batch'], CONFIG['height'], CONFIG['width'], CONFIG['features'])
    x = jax.random.uniform(key, shape, dtype)
    gain = jnp.ones((CONFIG['features'],), dtype)
    bias = jnp.zeros((CONFIG['features'],), dtype)
    return x, gain, bias


def workload(x, gain, bias):
    axis = CONFIG['axis']
    epsilon = CONFIG['epsilon']
    mu = jnp.mean(x, axis=axis, keepdims=True)
    sigma = jnp.sqrt(jnp.var(x, axis=axis, ddof=CONFIG['ddof'], keepdims=True) + epsilon)
    return gain * (x - mu) / (sigma + epsilon) + bias
