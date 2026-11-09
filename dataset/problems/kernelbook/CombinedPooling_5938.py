"""CombinedPooling: global max pool + global avg pool over H, W (AdaptiveMax/AvgPool2d(1)), summed."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'CombinedPooling_5938',
    'batch': 16,
    'channels': 256,
    'height': 128,
    'width': 128,
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    x = jax.random.uniform(
        key,
        (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width']),
        dtype,
    )
    return (x,)


def workload(x):
    max_pooled = jnp.max(x, axis=(2, 3), keepdims=True)
    avg_pooled = jnp.mean(x, axis=(2, 3), keepdims=True)
    return max_pooled + avg_pooled
