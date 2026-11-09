"""MaxPoolStride1: replicate pad (right/bottom by 1) followed by 2x2 max pool with stride 1."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'MaxPoolStride1_4763',
    'batch': 8,
    'channels': 128,
    'height': 256,
    'width': 256,
    'pad': 1,
    'kernel_size': 2,
    'stride': 1,
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    x = jax.random.uniform(
        key, (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width']), dtype
    )
    return (x,)


def workload(x):
    p = CONFIG['pad']
    k = CONFIG['kernel_size']
    s = CONFIG['stride']
    xp = jnp.pad(x, ((0, 0), (0, 0), (0, p), (0, p)), mode='edge')
    return lax.reduce_window(
        xp, -jnp.inf, lax.max, (1, 1, k, k), (1, 1, s, s), 'VALID'
    )
