"""MiniBatchAverageLayer: batch stddev -> channel mean -> broadcast -> concat on channel axis."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'MiniBatchAverageLayer_3290',
    'batch': 16,
    'channels': 256,
    'height': 128,
    'width': 128,
    'offset': 1e-8,
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
    mean = jnp.mean(x, axis=0, keepdims=True)
    var = jnp.mean((x - mean) ** 2, axis=0, keepdims=True)
    stddev = jnp.sqrt(var + CONFIG['offset'])
    inject = jnp.mean(stddev, axis=1, keepdims=True)
    n, _, h, w = x.shape
    inject = jnp.broadcast_to(inject, (n, 1, h, w))
    return jnp.concatenate((x, inject), axis=1)
