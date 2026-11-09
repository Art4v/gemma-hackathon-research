"""DRS: ReLU -> global spatial max (AdaptiveMaxPool2d(1)) -> min(x, delta * max)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'DRS_15991',
    'batch': 8,
    'channels': 128,
    'height': 256,
    'width': 256,
    'delta': 4.0,
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
    x = jnp.maximum(x, 0)
    x_max = jnp.max(x, axis=(2, 3), keepdims=True)
    return jnp.minimum(x, x_max * CONFIG['delta'])
