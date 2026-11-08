"""GEMM followed by LogSumExp, LeakyReLU, LeakyReLU, GELU, and GELU."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': '64_Gemm_LogSumExp_LeakyReLU_LeakyReLU_GELU_GELU',
    'batch_size': 1024,
    'in_features': 8192,
    'out_features': 8192,
    'negative_slope': 0.01,
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    k1, k2, k3 = jax.random.split(key, 3)
    B = CONFIG['batch_size']
    K = CONFIG['in_features']
    N = CONFIG['out_features']
    x = jax.random.uniform(k1, (B, K), dtype=dtype)
    weight = jax.random.normal(k2, (K, N), dtype=dtype) * 0.02
    bias = jax.random.normal(k3, (N,), dtype=dtype) * 0.02
    return x, weight, bias


def workload(x, weight, bias):
    slope = CONFIG['negative_slope']
    x = jnp.dot(x, weight) + bias
    x = jax.nn.logsumexp(x, axis=1, keepdims=True)
    x = jax.nn.leaky_relu(x, negative_slope=slope)
    x = jax.nn.leaky_relu(x, negative_slope=slope)
    x = jax.nn.gelu(x, approximate=False)
    x = jax.nn.gelu(x, approximate=False)
    return x
