"""Replicate pad (right/bottom by k-1) -> MaxPool2d(kernel=k, stride=k-1)."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'MaxPoolStride1_9509',
    'batch': 16,
    'channels': 64,
    'height': 256,
    'width': 256,
    'kernel_size': 4,
    'stride': 3,  # torch module passes kernel_size - 1 as MaxPool2d's stride
    'pad': 3,     # replicate pad on right and bottom only
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
    k = CONFIG['kernel_size']
    s = CONFIG['stride']
    p = CONFIG['pad']
    padded = jnp.pad(x, ((0, 0), (0, 0), (0, p), (0, p)), mode='edge')
    return lax.reduce_window(
        padded,
        jnp.array(-jnp.inf, dtype=x.dtype),
        lax.max,
        window_dimensions=(1, 1, k, k),
        window_strides=(1, 1, s, s),
        padding='VALID',
    )
