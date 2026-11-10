"""[bf16 variant] PixelNorm: x * rsqrt(mean(x**2, axis=channel) + eps)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'PixelNorm_9401',
    'batch': 8,
    'channels': 512,
    'height': 128,
    'width': 128,
    'eps': 1e-8,
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    x = jax.random.uniform(
        key,
        (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width']),
        dtype=dtype,
    )
    return (x,)


def workload(x):
    ms = jnp.mean(x * x, axis=1, keepdims=True)
    return x * jax.lax.rsqrt(ms + CONFIG['eps'])
