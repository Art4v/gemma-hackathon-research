"""[bf16 variant] Channel LayerNorm (NCHW, reduce over C): (x - mean) / (std + eps) * g + b."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'LayerNorm_9611',
    'batch': 16,
    'dim': 128,
    'height': 128,
    'width': 128,
    'eps': 1e-5,
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    x = jax.random.uniform(
        key, (CONFIG['batch'], CONFIG['dim'], CONFIG['height'], CONFIG['width']), dtype)
    g = jnp.ones((1, CONFIG['dim'], 1, 1), dtype)
    b = jnp.zeros((1, CONFIG['dim'], 1, 1), dtype)
    return x, g, b


def workload(x, g, b):
    mean = jnp.mean(x, axis=1, keepdims=True)
    var = jnp.mean(jnp.square(x - mean), axis=1, keepdims=True)
    std = jnp.sqrt(var)
    return (x - mean) / (std + CONFIG['eps']) * g + b
