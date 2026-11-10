"""[bf16 variant] PixelNorm: x * rsqrt(mean(x**2, axis=dim, keepdims=True) + eps)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'PixelNorm_17064',
    'shape': (8, 32, 32, 64, 128),
    'dim': 4,
    'eps': 1e-8,
}


def create_inputs(dtype=jnp.bfloat16):
    x = jax.random.uniform(jax.random.key(0), CONFIG['shape'], dtype)
    return (x,)


def workload(x):
    ms = jnp.mean(x * x, axis=CONFIG['dim'], keepdims=True)
    return x * jax.lax.rsqrt(ms + CONFIG['eps'])
