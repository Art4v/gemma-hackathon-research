"""[bf16 variant] PixelNorm: x / sqrt(mean(x**2 over channels) + eps)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'PixelNorm_16870',
    'batch': 16,
    'channels': 256,
    'height': 128,
    'width': 128,
    'eps': 1e-8,
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    x = jax.random.uniform(
        key,
        (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width']),
        dtype,
    )
    return (x,)


def workload(x):
    ms = jnp.mean(x * x, axis=1, keepdims=True)
    return x / jnp.sqrt(ms + CONFIG['eps'])
