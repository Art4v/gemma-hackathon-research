"""[bf16 variant] PixelNorm: x * rsqrt(mean(x**2, channel axis) + eps)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'PixelNorm_4790',
    'batch': 8,
    'channels': 512,
    'height': 128,
    'width': 128,
    'eps': 1e-8,
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    shape = (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width'])
    x = jax.random.uniform(key, shape, dtype)
    return (x,)


def workload(x):
    ms = jnp.mean(x * x, axis=1, keepdims=True)
    return x * lax_rsqrt(ms + CONFIG['eps'])


def lax_rsqrt(v):
    return jax.lax.rsqrt(v)
