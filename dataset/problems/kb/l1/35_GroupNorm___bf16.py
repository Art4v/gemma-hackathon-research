"""[bf16 variant] Group Normalization over 4D NCHW tensor."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': '35_GroupNorm_',
    'batch_size': 8,
    'features': 128,
    'num_groups': 8,
    'dim1': 256,
    'dim2': 256,
    'eps': 1e-5,
}

def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    B = CONFIG['batch_size']
    C = CONFIG['features']
    H = CONFIG['dim1']
    W = CONFIG['dim2']
    x = jax.random.uniform(key, (B, C, H, W), dtype=dtype)
    weight = jnp.ones((C,), dtype=dtype)
    bias = jnp.zeros((C,), dtype=dtype)
    return x, weight, bias

def workload(x, weight, bias):
    B, C, H, W = x.shape
    G = CONFIG['num_groups']
    eps = CONFIG['eps']
    x_reshaped = jnp.reshape(x, (B, G, C // G, H, W))
    mean = jnp.mean(x_reshaped, axis=(2, 3, 4), keepdims=True)
    var = jnp.var(x_reshaped, axis=(2, 3, 4), keepdims=True)
    x_norm = (x_reshaped - mean) / jnp.sqrt(var + eps)
    x_norm = jnp.reshape(x_norm, (B, C, H, W))
    return x_norm * jnp.reshape(weight, (1, C, 1, 1)) + jnp.reshape(bias, (1, C, 1, 1))
