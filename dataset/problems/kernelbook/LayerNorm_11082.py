"""[bf16 variant] LayerNorm over (C, H) axes of an NCHW tensor with per-channel affine: (x - mean) / sqrt(var + eps) * w[c] + b[c]."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'LayerNorm_11082',
    'batch': 8,
    'num_channels': 128,
    'height': 256,
    'width': 256,
    'eps': 1e-5,
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    shape = (CONFIG['batch'], CONFIG['num_channels'], CONFIG['height'], CONFIG['width'])
    x = jax.random.uniform(key, shape, dtype)
    weight = jnp.ones((CONFIG['num_channels'],), dtype)
    bias = jnp.zeros((CONFIG['num_channels'],), dtype)
    return x, weight, bias


def workload(x, weight, bias):
    mu = jnp.mean(x, axis=(1, 2), keepdims=True)
    sig = jnp.sqrt(jnp.mean((x - mu) ** 2, axis=(1, 2), keepdims=True) + CONFIG['eps'])
    return (x - mu) / sig * weight[:, None, None] + bias[:, None, None]
