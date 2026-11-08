"""GEMM followed by GroupNorm and Hardtanh."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': '30_Gemm_GroupNorm_Hardtanh',
    'batch_size': 1024,
    'in_features': 8192,
    'out_features': 8192,
    'num_groups': 16,
    'hardtanh_min': -2.0,
    'hardtanh_max': 2.0,
    'eps': 1e-5,
}

def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    k1, k2, k3 = jax.random.split(key, 3)
    B = CONFIG['batch_size']
    I = CONFIG['in_features']
    O = CONFIG['out_features']

    x = jax.random.uniform(k1, (B, I), dtype=dtype)
    weight = jax.random.normal(k2, (I, O), dtype=dtype) * 0.02
    bias = jax.random.normal(k3, (O,), dtype=dtype) * 0.02
    gn_weight = jnp.ones((O,), dtype=dtype)
    gn_bias = jnp.zeros((O,), dtype=dtype)

    return x, weight, bias, gn_weight, gn_bias

def workload(x, weight, bias, gn_weight, gn_bias):
    B, O = x.shape[0], weight.shape[1]
    G = CONFIG['num_groups']
    eps = CONFIG['eps']

    x = jnp.dot(x, weight) + bias

    x_reshaped = jnp.reshape(x, (B, G, O // G))
    mean = jnp.mean(x_reshaped, axis=-1, keepdims=True)
    var = jnp.var(x_reshaped, axis=-1, keepdims=True)
    x_norm = (x_reshaped - mean) / jnp.sqrt(var + eps)
    x = jnp.reshape(x_norm, (B, O)) * gn_weight + gn_bias

    x = jnp.clip(x, CONFIG['hardtanh_min'], CONFIG['hardtanh_max'])
    return x
