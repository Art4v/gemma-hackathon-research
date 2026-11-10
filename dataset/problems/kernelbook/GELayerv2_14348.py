"""GELayerv2: global average pool over spatial axes -> sigmoid -> multiply with input -> residual add."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'GELayerv2_14348',
    'batch': 8,
    'channels': 128,
    'height': 256,
    'width': 256,
    'spatial_axes': (2, 3),
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    shape = (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width'])
    x = jax.random.uniform(key, shape, dtype)
    return (x,)


def workload(x):
    y = jnp.mean(x, axis=CONFIG['spatial_axes'], keepdims=True)
    y = jax.nn.sigmoid(y)
    z = x * y
    return x + z
