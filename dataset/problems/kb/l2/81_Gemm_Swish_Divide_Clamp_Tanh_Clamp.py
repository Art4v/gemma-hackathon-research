"""GEMM followed by Swish, divide, clamp, tanh, and clamp."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': '81_Gemm_Swish_Divide_Clamp_Tanh_Clamp',
    'batch_size': 1024,
    'in_features': 8192,
    'out_features': 8192,
    'div_factor': 2.0,
    'clamp_min': -1.0,
    'clamp_max': 1.0,
}

def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    k1, k2, k3 = jax.random.split(key, 3)
    batch_size = CONFIG['batch_size']
    in_features = CONFIG['in_features']
    out_features = CONFIG['out_features']
    x = jax.random.uniform(k1, (batch_size, in_features), dtype=dtype)
    w = jax.random.normal(k2, (in_features, out_features), dtype=dtype) * 0.02
    b = jax.random.normal(k3, (out_features,), dtype=dtype) * 0.02
    return (x, w, b)

def workload(x, w, b):
    x = jnp.dot(x, w) + b
    x = x * jax.nn.sigmoid(x)
    x = x / CONFIG['div_factor']
    x = jnp.clip(x, CONFIG['clamp_min'], CONFIG['clamp_max'])
    x = jnp.tanh(x)
    x = jnp.clip(x, CONFIG['clamp_min'], CONFIG['clamp_max'])
    return x
