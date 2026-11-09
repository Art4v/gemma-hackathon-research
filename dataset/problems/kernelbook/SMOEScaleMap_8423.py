"""SMOE scale map: x + eps -> mean over channels, log2 -> (log2(mean) - mean(log2)) * mean."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'SMOEScaleMap_8423',
    'batch': 16,
    'channels': 128,
    'height': 128,
    'width': 256,
    'eps': 1e-7,
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
    x = x + CONFIG['eps']
    m = jnp.mean(x, axis=1)
    k = jnp.log2(m) - jnp.mean(jnp.log2(x), axis=1)
    return k * m
