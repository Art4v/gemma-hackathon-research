"""LayerNormChannel: per-pixel normalization over the channel axis of NCHW, then per-channel affine."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'LayerNormChannel_13867',
    'batch': 16,
    'channels': 128,
    'height': 128,
    'width': 128,
    'eps': 1e-5,
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    x = jax.random.uniform(
        key,
        (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width']),
        dtype,
    )
    weight = jnp.ones((CONFIG['channels'],), dtype)
    bias = jnp.zeros((CONFIG['channels'],), dtype)
    return x, weight, bias


def workload(x, weight, bias):
    u = jnp.mean(x, axis=1, keepdims=True)
    d = x - u
    s = jnp.mean(d * d, axis=1, keepdims=True)
    y = d / jnp.sqrt(s + CONFIG['eps'])
    return weight[None, :, None, None] * y + bias[None, :, None, None]
