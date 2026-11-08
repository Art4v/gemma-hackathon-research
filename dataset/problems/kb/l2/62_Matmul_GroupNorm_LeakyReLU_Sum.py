"""Matmul, GroupNorm, LeakyReLU, and element-wise sum."""

import jax
import jax.numpy as jnp

CONFIG = {
    'name': '62_Matmul_GroupNorm_LeakyReLU_Sum',
    'batch_size': 1024,
    'input_size': 8192,
    'hidden_size': 8192,
    'num_groups': 512,
    'eps': 1e-5,
    'negative_slope': 0.01,
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    k1, k2, k3 = jax.random.split(key, 3)
    batch_size = CONFIG['batch_size']
    input_size = CONFIG['input_size']
    hidden_size = CONFIG['hidden_size']

    x = jax.random.uniform(k1, (batch_size, input_size), dtype=dtype)
    fc_weight = jax.random.normal(k2, (input_size, hidden_size), dtype=dtype) * 0.02
    fc_bias = jax.random.normal(k3, (hidden_size,), dtype=dtype) * 0.02
    gn_weight = jnp.ones((hidden_size,), dtype=dtype)
    gn_bias = jnp.zeros((hidden_size,), dtype=dtype)
    return (x, fc_weight, fc_bias, gn_weight, gn_bias)


def workload(x, fc_weight, fc_bias, gn_weight, gn_bias):
    batch_size, _ = x.shape
    hidden_size = fc_weight.shape[1]
    num_groups = CONFIG['num_groups']
    eps = CONFIG['eps']
    negative_slope = CONFIG['negative_slope']

    x = jnp.dot(x, fc_weight) + fc_bias
    x_grouped = jnp.reshape(x, (batch_size, num_groups, hidden_size // num_groups))
    mean = jnp.mean(x_grouped, axis=-1, keepdims=True)
    var = jnp.var(x_grouped, axis=-1, keepdims=True)
    x_norm = (x_grouped - mean) / jnp.sqrt(var + eps)
    x = jnp.reshape(x_norm, (batch_size, hidden_size)) * gn_weight + gn_bias
    x = jax.nn.leaky_relu(x, negative_slope=negative_slope)
    x = x + x
    return x
