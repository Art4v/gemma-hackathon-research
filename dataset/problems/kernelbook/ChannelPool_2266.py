"""ChannelPool: concat(max over channels, mean over channels) along dim 1."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'ChannelPool_2266',
    'batch': 16,
    'channels': 64,
    'height': 256,
    'width': 256,
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    x = jax.random.uniform(
        key,
        (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width']),
        dtype=dtype,
    )
    return (x,)


def workload(x):
    mx = jnp.max(x, axis=1, keepdims=True)
    mean = jnp.mean(x, axis=1, keepdims=True)
    return jnp.concatenate((mx, mean), axis=1)
