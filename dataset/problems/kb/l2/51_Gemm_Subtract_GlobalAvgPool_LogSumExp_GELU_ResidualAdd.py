"""Gemm, Subtract, GlobalAvgPool, LogSumExp, GELU, and ResidualAdd."""

import jax
import jax.numpy as jnp

CONFIG = {
    'name': '51_Gemm_Subtract_GlobalAvgPool_LogSumExp_GELU_ResidualAdd',
    'batch_size': 2048,
    'in_features': 8192,
    'out_features': 8192,
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    k1, k2, k3, k4 = jax.random.split(key, 4)
    batch_size = CONFIG['batch_size']
    in_features = CONFIG['in_features']
    out_features = CONFIG['out_features']

    x = jax.random.uniform(k1, (batch_size, in_features), dtype=dtype)
    weight = jax.random.normal(k2, (in_features, out_features), dtype=dtype) * 0.02
    bias = jax.random.normal(k3, (out_features,), dtype=dtype) * 0.02
    subtract = jax.random.normal(k4, (out_features,), dtype=dtype) * 0.02
    return x, weight, bias, subtract


def workload(x, weight, bias, subtract):
    original_x = x
    x = jnp.dot(x, weight) + bias
    x = x - subtract
    x = jnp.mean(x, axis=1, keepdims=True)
    x = jax.scipy.special.logsumexp(x, axis=1, keepdims=True)
    x = jax.nn.gelu(x, approximate=False)
    x = x + original_x
    return x
