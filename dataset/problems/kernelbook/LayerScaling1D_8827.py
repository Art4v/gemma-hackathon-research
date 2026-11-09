"""LayerScaling1D: x / sqrt(mean(x^2, axis=1) + eps)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'LayerScaling1D_8827',
    'batch': 16,
    'channels': 256,
    'height': 128,
    'width': 128,
    'eps': 1e-5,
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    x = jax.random.uniform(
        key,
        (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width']),
        dtype=dtype,
    )
    return (x,)


def workload(x):
    moment2 = jnp.mean(x * x, axis=1, keepdims=True)
    return x / jnp.sqrt(moment2 + CONFIG['eps'])
