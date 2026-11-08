"""Matrix multiplication, elementwise scaling, and 1D batch normalization."""

import jax
import jax.numpy as jnp

CONFIG = {
    'name': '39_Gemm_Scale_BatchNorm',
    'batch_size': 16384,
    'in_features': 4096,
    'out_features': 4096,
    'eps': 1e-5,
}

def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    k1, k2, k3, k4 = jax.random.split(key, 4)
    B = CONFIG['batch_size']
    I = CONFIG['in_features']
    O = CONFIG['out_features']
    
    x = jax.random.uniform(k1, (B, I), dtype=dtype)
    weight = jax.random.normal(k2, (I, O), dtype=dtype) * 0.02
    bias = jax.random.normal(k3, (O,), dtype=dtype) * 0.02
    scale = jax.random.normal(k4, (O,), dtype=dtype)
    bn_weight = jnp.ones((O,), dtype=dtype)
    bn_bias = jnp.zeros((O,), dtype=dtype)
    return x, weight, bias, scale, bn_weight, bn_bias

def workload(x, weight, bias, scale, bn_weight, bn_bias):
    x = jnp.dot(x, weight) + bias
    x = x * scale
    mean = jnp.mean(x, axis=0, keepdims=True)
    var = jnp.var(x, axis=0, keepdims=True)
    x = (x - mean) / jnp.sqrt(var + CONFIG['eps'])
    x = x * bn_weight + bn_bias
    return x
