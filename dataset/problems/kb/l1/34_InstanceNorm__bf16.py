"""[bf16 variant] InstanceNorm2d operation across spatial dimensions (H, W) per batch and channel."""

import jax
import jax.numpy as jnp

CONFIG = {
    'name': '34_InstanceNorm',
    'batch_size': 16,
    'num_features': 64,
    'height': 256,
    'width': 256,
    'eps': 1e-5,
}

def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    shape = (
        CONFIG['batch_size'],
        CONFIG['num_features'],
        CONFIG['height'],
        CONFIG['width'],
    )
    x = jax.random.uniform(key, shape=shape, dtype=dtype)
    return (x,)

def workload(x):
    mean = jnp.mean(x, axis=(2, 3), keepdims=True)
    var = jnp.var(x, axis=(2, 3), keepdims=True)
    return (x - mean) / jnp.sqrt(var + CONFIG['eps'])
