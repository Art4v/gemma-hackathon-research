"""LayerNormChan: per-pixel LayerNorm over the channel axis of an NCHW tensor, with scale g and shift b."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'LayerNormChan_10748',
    'batch': 16,
    'dim': 256,
    'height': 128,
    'width': 128,
    'eps': 1e-5,
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    x = jax.random.uniform(
        key,
        (CONFIG['batch'], CONFIG['dim'], CONFIG['height'], CONFIG['width']),
        dtype,
    )
    g = jnp.ones((1, CONFIG['dim'], 1, 1), dtype)
    b = jnp.zeros((1, CONFIG['dim'], 1, 1), dtype)
    return x, g, b


def workload(x, g, b):
    mean = jnp.mean(x, axis=1, keepdims=True)
    var = jnp.var(x, axis=1, keepdims=True)  # biased, matches unbiased=False
    return (x - mean) / jnp.sqrt(var + CONFIG['eps']) * g + b
