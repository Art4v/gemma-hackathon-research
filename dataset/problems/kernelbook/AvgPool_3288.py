"""AvgPool_3288: zero pad by 1 on H and W, then a 3x3 average pool with stride 2 (padded zeros count toward the mean)."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'AvgPool_3288',
    'batch': 8,
    'channels': 128,
    'height': 256,
    'width': 256,
    'pad': 1,
    'kernel_size': 3,
    'stride': 2,
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
    p = CONFIG['pad']
    k = CONFIG['kernel_size']
    s = CONFIG['stride']
    xp = jnp.pad(x, ((0, 0), (0, 0), (p, p), (p, p)))
    summed = lax.reduce_window(
        xp,
        jnp.array(0, dtype=x.dtype),
        lax.add,
        window_dimensions=(1, 1, k, k),
        window_strides=(1, 1, s, s),
        padding='VALID',
    )
    return summed / jnp.array(k * k, dtype=x.dtype)
