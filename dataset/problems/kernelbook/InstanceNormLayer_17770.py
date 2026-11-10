"""Instance normalization over NCHW spatial axes: spatial mean subtraction, spatial RMS division with eps."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'InstanceNormLayer_17770',
    'batch': 8,
    'channels': 128,
    'height': 256,
    'width': 256,
    'eps': 1e-08,
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    shape = (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width'])
    x = jax.random.uniform(key, shape, dtype)
    return (x,)


def workload(x):
    x = x - jnp.mean(x, axis=(2, 3), keepdims=True)
    x = x / jnp.sqrt(jnp.mean(x ** 2, axis=(2, 3), keepdims=True) + CONFIG['eps'])
    return x
