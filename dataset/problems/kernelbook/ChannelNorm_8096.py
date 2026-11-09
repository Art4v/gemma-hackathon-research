"""ChannelNorm: x / (max over batch and length of |x|, per channel + eps)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'ChannelNorm_8096',
    'batch': 16,
    'channels': 512,
    'length': 8192,
    'eps': 1e-05,
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    x = jax.random.uniform(
        key, (CONFIG['batch'], CONFIG['channels'], CONFIG['length']), dtype)
    return (x,)


def workload(x):
    divider = jnp.max(jnp.abs(x), axis=(0, 2), keepdims=True) + CONFIG['eps']
    return x / divider
