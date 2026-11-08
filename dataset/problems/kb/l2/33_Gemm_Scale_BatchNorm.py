"""GEMM followed by elementwise scaling and 1D Batch Normalization."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': '33_Gemm_Scale_BatchNorm',
    'batch_size': 1024,
    'in_features': 8192,
    'out_features': 8192,
    'eps': 1e-5,
}

def create_inputs(dtype=jnp.float32):
    k0, k1, k2, k3 = jax.random.split(jax.random.key(0), 4)
    B = CONFIG['batch_size']
    I = CONFIG['in_features']
    O = CONFIG['out_features']
    x = jax.random.uniform(k0, (B, I), dtype=dtype)
    w = jax.random.normal(k1, (I, O), dtype=dtype) * 0.02
    b = jax.random.normal(k2, (O,), dtype=dtype) * 0.02
    scale = jax.random.normal(k3, (O,), dtype=dtype)
    bn_w = jnp.ones((O,), dtype=dtype)
    bn_b = jnp.zeros((O,), dtype=dtype)
    return (x, w, b, scale, bn_w, bn_b)

def workload(x, w, b, scale, bn_w, bn_b):
    x = jnp.dot(x, w) + b
    x = x * scale
    mean = jnp.mean(x, axis=0, keepdims=True)
    var = jnp.var(x, axis=0, keepdims=True)
    x = (x - mean) / jnp.sqrt(var + CONFIG['eps']) * bn_w + bn_b
    return x
