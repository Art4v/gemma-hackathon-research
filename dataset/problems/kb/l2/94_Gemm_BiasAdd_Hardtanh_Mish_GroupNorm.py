"""GEMM, BiasAdd, Hardtanh, Mish, and GroupNorm in sequence."""

import jax
import jax.numpy as jnp

CONFIG = {
    'name': '94_Gemm_BiasAdd_Hardtanh_Mish_GroupNorm',
    'batch_size': 1024,
    'in_features': 8192,
    'out_features': 8192,
    'num_groups': 256,
    'hardtanh_min': -1.0,
    'hardtanh_max': 1.0,
    'eps': 1e-5,
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    k1, k2, k3, k4 = jax.random.split(key, 4)
    B = CONFIG['batch_size']
    K = CONFIG['in_features']
    N = CONFIG['out_features']

    x = jax.random.uniform(k1, (B, K), dtype=dtype)
    w = jax.random.normal(k2, (K, N), dtype=dtype) * 0.02
    b_gemm = jax.random.normal(k3, (N,), dtype=dtype) * 0.02
    b_extra = jax.random.normal(k4, (N,), dtype=dtype) * 0.02
    gn_weight = jnp.ones((N,), dtype=dtype)
    gn_bias = jnp.zeros((N,), dtype=dtype)

    return x, w, b_gemm, b_extra, gn_weight, gn_bias


def workload(x, w, b_gemm, b_extra, gn_weight, gn_bias):
    B, _ = x.shape
    N = w.shape[1]
    G = CONFIG['num_groups']

    # GEMM + BiasAdd
    x = jnp.dot(x, w) + b_gemm
    x = x + b_extra

    # Hardtanh
    x = jnp.clip(x, CONFIG['hardtanh_min'], CONFIG['hardtanh_max'])

    # Mish
    x = x * jnp.tanh(jax.nn.softplus(x))

    # GroupNorm
    x = x.reshape(B, G, N // G)
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.var(x, axis=-1, keepdims=True)
    x = (x - mean) / jnp.sqrt(var + CONFIG['eps'])
    x = x.reshape(B, N)
    x = x * gn_weight + gn_bias

    return x
