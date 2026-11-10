"""KDE_7676: L2 normalize two embeddings along dim 1, then sum-reduced MSE between them."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'KDE_7676',
    'batch': 8,
    'channels': 128,
    'height': 128,
    'width': 256,
    'norm_axis': 1,
    'eps': 1e-12,
}


def create_inputs(dtype=jnp.float32):
    shape = (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width'])
    key = jax.random.key(0)
    k_s, k_t = jax.random.split(key)
    embedding_s = jax.random.uniform(k_s, shape, dtype)
    embedding_t = jax.random.uniform(k_t, shape, dtype)
    return embedding_s, embedding_t


def _normalize(x):
    norm = jnp.sqrt(jnp.sum(x * x, axis=CONFIG['norm_axis'], keepdims=True))
    return x / jnp.maximum(norm, CONFIG['eps'])


def workload(embedding_s, embedding_t):
    inputs_embed = _normalize(embedding_s)
    targets_embed = _normalize(embedding_t)
    return jnp.sum((inputs_embed - targets_embed) ** 2)
