"""GEMM followed by sigmoid, scaling, and residual addition."""

import jax
import jax.numpy as jnp

CONFIG = {
    'name': '70_Gemm_Sigmoid_Scaling_ResidualAdd',
    'batch_size': 1024,
    'input_size': 8192,
    'hidden_size': 8192,
    'scaling_factor': 2.0,
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    k1, k2, k3 = jax.random.split(key, 3)
    batch_size = CONFIG['batch_size']
    input_size = CONFIG['input_size']
    hidden_size = CONFIG['hidden_size']

    x = jax.random.uniform(k1, (batch_size, input_size), dtype=dtype)
    w = jax.random.normal(k2, (input_size, hidden_size), dtype=dtype) * 0.02
    b = jax.random.normal(k3, (hidden_size,), dtype=dtype) * 0.02
    return (x, w, b)


def workload(x, w, b):
    scaling_factor = CONFIG['scaling_factor']
    x = jnp.dot(x, w) + b
    original_x = x
    x = jax.nn.sigmoid(x)
    x = x * scaling_factor
    x = x + original_x
    return x
