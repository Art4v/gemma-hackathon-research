"""Global average pool (AdaptiveAvgPool2d(1,1)) -> flatten -> Linear."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'EmbeddingModule_16872',
    'batch': 8,
    'in_channels': 512,
    'height': 128,
    'width': 128,
    'desc_channels': 1024,
}


def create_inputs(dtype=jnp.float32):
    kx, kw, kb = jax.random.split(jax.random.key(0), 3)
    x = jax.random.uniform(
        kx, (CONFIG['batch'], CONFIG['in_channels'], CONFIG['height'], CONFIG['width']), dtype)
    w = jax.random.normal(kw, (CONFIG['in_channels'], CONFIG['desc_channels']), dtype) * 0.02
    b = jax.random.normal(kb, (CONFIG['desc_channels'],), dtype) * 0.02
    return x, w, b


def workload(x, w, b):
    pooled = jnp.mean(x, axis=(2, 3))  # (B, C), NCHW global average pool
    return pooled @ w + b
