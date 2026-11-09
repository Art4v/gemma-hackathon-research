"""ChanNorm: per-pixel normalization over channels, (x - mean) / (std + eps) * g + b."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'ChanNorm_5651',
    'batch': 16,
    'dim': 128,
    'height': 128,
    'width': 128,
    'eps': 1e-5,
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    x = jax.random.uniform(
        key, (CONFIG['batch'], CONFIG['dim'], CONFIG['height'], CONFIG['width']), dtype)
    g = jnp.ones((1, CONFIG['dim'], 1, 1), dtype)
    b = jnp.zeros((1, CONFIG['dim'], 1, 1), dtype)
    return x, g, b


def workload(x, g, b):
    mean = jnp.mean(x, axis=1, keepdims=True)
    # Biased variance; eps is added to std (not inside the sqrt), matching the torch module.
    std = jnp.sqrt(jnp.var(x, axis=1, keepdims=True))
    return (x - mean) / (std + CONFIG['eps']) * g + b
