"""[bf16 variant] DiffLoss: flatten, detached L2 row-normalize, GEMM D1 @ D2^T, mean of squares."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'DiffLoss_14239',
    'batch': 32,
    'channels': 128,
    'height': 128,
    'width': 128,
    'eps': 1e-6,
}


def create_inputs(dtype=jnp.bfloat16):
    shape = (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width'])
    key = jax.random.key(0)
    k1, k2 = jax.random.split(key)
    d1 = jax.random.uniform(k1, shape, dtype)
    d2 = jax.random.uniform(k2, shape, dtype)
    return d1, d2


def _normalize_rows(x):
    flat = x.reshape(x.shape[0], -1)
    norm = lax.stop_gradient(jnp.linalg.norm(flat, ord=2, axis=1, keepdims=True))
    return flat / (norm + CONFIG['eps'])


def workload(d1, d2):
    d1_norm = _normalize_rows(d1)
    d2_norm = _normalize_rows(d2)
    return jnp.mean(jnp.square(d1_norm @ d2_norm.T))
