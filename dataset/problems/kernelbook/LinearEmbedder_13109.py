"""Linear (x @ w + b) followed by row-wise L2 normalization (LinearEmbedder)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'LinearEmbedder_13109',
    'batch': 16384,
    'in_features': 2048,
    'out_features': 2048,
    'eps': 1e-12,
}


def create_inputs(dtype=jnp.float32):
    kx, kw, kb = jax.random.split(jax.random.key(0), 3)
    x = jax.random.uniform(kx, (CONFIG['batch'], CONFIG['in_features']), dtype)
    w = jax.random.normal(kw, (CONFIG['in_features'], CONFIG['out_features']), dtype) * 0.02
    b = jax.random.normal(kb, (CONFIG['out_features'],), dtype) * 0.02
    return x, w, b


def workload(x, w, b):
    o = x @ w + b
    # eps goes inside the sqrt, matching torch.sum(...).add_(1e-12) before sqrt.
    norm = jnp.sqrt(jnp.sum(o * o, axis=1, keepdims=True) + CONFIG['eps'])
    return o / norm
