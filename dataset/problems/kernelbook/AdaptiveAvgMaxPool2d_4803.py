"""AdaptiveAvgPool2d(output_size=1): global average pool over the spatial axes of an NCHW tensor."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'AdaptiveAvgMaxPool2d_4803',
    'batch': 8,
    'channels': 128,
    'height': 256,
    'width': 256,
    'output_size': 1,
    'pool_type': 'avg',
    'spatial_axes': (2, 3),
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
    return jnp.mean(x, axis=CONFIG['spatial_axes'], keepdims=True)
