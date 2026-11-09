"""Replicate pad (right/bottom by k-1) followed by MaxPool2d(kernel=k, stride=k-1)."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'MaxPoolStride1_11689',
    'batch': 16,
    'channels': 64,
    'height': 256,
    'width': 256,
    'kernel_size': 4,
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
    pad = k - 1
    # F.pad(..., (0, pad, 0, pad), mode='replicate') pads W and H on the high side only.
    padded = jnp.pad(x, ((0, 0), (0, 0), (0, pad), (0, pad)), mode='edge')
    # The torch module passes self.pad as MaxPool2d's stride, so stride = k - 1.
    return lax.reduce_window(
        padded,
        jnp.array(-jnp.inf, dtype=x.dtype),
        lax.max,
        window_dimensions=(1, 1, k, k),
        window_strides=(1, 1, pad, pad),
        padding='VALID',
    )
