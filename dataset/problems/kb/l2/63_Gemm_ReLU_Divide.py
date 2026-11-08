"""Fused GEMM, ReLU, and scalar division."""

import jax
import jax.numpy as jnp

CONFIG = {
    'name': '63_Gemm_ReLU_Divide',
    'batch_size': 1024,
    'in_features': 8192,
    'out_features': 8192,
    'divisor': 2.0,
}

def create_inputs(dtype=jnp.float32):
    k1, k2, k3 = jax.random.split(jax.random.key(0), 3)
    batch_size = CONFIG['batch_size']
    in_features = CONFIG['in_features']
    out_features = CONFIG['out_features']
    
    x = jax.random.uniform(k1, (batch_size, in_features), dtype=dtype)
    w = jax.random.normal(k2, (in_features, out_features), dtype=dtype) * 0.02
    b = jax.random.normal(k3, (out_features,), dtype=dtype) * 0.02
    return (x, w, b)

def workload(x, w, b):
    out = jnp.dot(x, w) + b
    out = jnp.maximum(out, 0)
    out = out / CONFIG['divisor']
    return out
