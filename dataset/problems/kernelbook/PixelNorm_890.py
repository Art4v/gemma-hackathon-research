"""[bf16 variant] PixelNorm: x / sqrt(mean(x ** 2, axis=0, keepdims=True) + eps)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'PixelNorm_890',
    'batch': 8,
    'channels': 128,
    'height': 256,
    'width': 256,
    'axis': 0,
    'eps': 1e-08,
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    shape = (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width'])
    x = jax.random.uniform(key, shape, dtype)
    return (x,)


def workload(x):
    mean_sq = jnp.mean(jnp.square(x), axis=CONFIG['axis'], keepdims=True)
    return x / jnp.sqrt(mean_sq + CONFIG['eps'])
