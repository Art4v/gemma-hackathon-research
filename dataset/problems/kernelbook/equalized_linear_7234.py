"""Equalized linear: scale input by gain/sqrt(fan_in), bias-free matmul, add bias."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'equalized_linear_7234',
    'batch': 16,
    'seq_len': 1024,
    'c_in': 2048,
    'c_out': 2048,
    'a': 1.0,
}


def create_inputs(dtype=jnp.float32):
    k_x, k_w = jax.random.split(jax.random.key(0))
    x = jax.random.uniform(k_x, (CONFIG['batch'], CONFIG['seq_len'], CONFIG['c_in']), dtype)
    w = jax.random.normal(k_w, (CONFIG['c_in'], CONFIG['c_out']), dtype) * 0.02
    bias = jnp.zeros((CONFIG['c_out'],), dtype)
    return x, w, bias


def workload(x, w, bias):
    fan_in = w.shape[0]
    gain = (2.0 / (1.0 + CONFIG['a'] ** 2)) ** 0.5
    scale = gain / fan_in ** 0.5
    return (x * scale) @ w + bias
