"""Gemm -> LogSumExp(dim=1, keepdim) -> LeakyReLU -> LeakyReLU -> GELU -> GELU."""
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
    kx, kw, kb = jax.random.split(jax.random.key(0), 3)
    x = jax.random.uniform(kx, (CONFIG['batch_size'], CONFIG['in_features']), dtype)
    w = jax.random.normal(kw, (CONFIG['in_features'], CONFIG['out_features']), dtype) * 0.02
    b = jax.random.normal(kb, (CONFIG['out_features'],), dtype) * 0.02
    return (x, w, b)


def workload(x, w, b):
    slope = CONFIG['negative_slope']
    y = x @ w + b
    y = jax.nn.logsumexp(y, axis=1, keepdims=True)
    y = jax.nn.leaky_relu(y, negative_slope=slope)
    y = jax.nn.leaky_relu(y, negative_slope=slope)
    y = jax.nn.gelu(y, approximate=False)
    y = jax.nn.gelu(y, approximate=False)
    return y
