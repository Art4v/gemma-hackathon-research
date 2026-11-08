"""Layer Normalization over 3D feature map (features, dim1, dim2)."""

import jax
import jax.numpy as jnp

CONFIG = {
    'name': '40_LayerNorm',
    'batch_size': 8,
    'features': 128,
    'dim1': 256,
    'dim2': 256,
    'eps': 1e-5,
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    b = CONFIG['batch_size']
    c = CONFIG['features']
    d1 = CONFIG['dim1']
    d2 = CONFIG['dim2']
    x = jax.random.uniform(key, (b, c, d1, d2), dtype=dtype)
    weight = jnp.ones((c, d1, d2), dtype=dtype)
    bias = jnp.zeros((c, d1, d2), dtype=dtype)
    return x, weight, bias


def workload(x, weight, bias):
    orig_dtype = x.dtype
    x_f32 = x.astype(jnp.float32)
    mean = jnp.mean(x_f32, axis=(-3, -2, -1), keepdims=True)
    var = jnp.var(x_f32, axis=(-3, -2, -1), keepdims=True)
    x_norm = (x_f32 - mean) * jax.lax.rsqrt(var + CONFIG['eps'])
    out = x_norm.astype(orig_dtype) * weight + bias
    return out
