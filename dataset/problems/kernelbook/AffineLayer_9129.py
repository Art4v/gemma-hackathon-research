"""AffineLayer_9129: per-channel scale of an NCHW tensor (broadcast weight, multiply)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'AffineLayer_9129',
    'batch': 8,
    'num_channels': 128,
    'height': 256,
    'width': 256,
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    x = jax.random.uniform(
        key,
        (CONFIG['batch'], CONFIG['num_channels'], CONFIG['height'], CONFIG['width']),
        dtype,
    )
    weight = jnp.ones((1, CONFIG['num_channels'], 1, 1), dtype)
    return x, weight


def workload(x, weight):
    return x * jnp.broadcast_to(weight, x.shape)
