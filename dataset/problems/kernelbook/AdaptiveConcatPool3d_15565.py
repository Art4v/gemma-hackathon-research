"""[bf16 variant] AdaptiveConcatPool3d: global 3D average pool and global 3D max pool over the last three axes, concatenated on axis 1."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'AdaptiveConcatPool3d_15565',
    'channels': 32,
    'depth': 128,
    'height': 128,
    'width': 128,
    'pool_axes': (-3, -2, -1),
    'concat_axis': 1,
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    shape = (CONFIG['channels'], CONFIG['depth'], CONFIG['height'], CONFIG['width'])
    x = jax.random.uniform(key, shape, dtype)
    return (x,)


def workload(x):
    axes = CONFIG['pool_axes']
    avg = jnp.mean(x, axis=axes, keepdims=True)
    max_ = jnp.max(x, axis=axes, keepdims=True)
    return jnp.concatenate((avg, max_), axis=CONFIG['concat_axis'])
