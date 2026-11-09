"""PixelNorm: x * rsqrt(mean(x**2, axis=2, keepdims=True) + eps)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'PixelNorm_4480',
    'batch': 16,
    'dim1': 256,
    'dim2': 128,
    'dim3': 128,
    'axis': 2,
    'eps': 1e-8,
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    shape = (CONFIG['batch'], CONFIG['dim1'], CONFIG['dim2'], CONFIG['dim3'])
    x = jax.random.uniform(key, shape, dtype)
    return (x,)


def workload(x):
    ms = jnp.mean(x * x, axis=CONFIG['axis'], keepdims=True)
    return x * jax.lax.rsqrt(ms + CONFIG['eps'])
