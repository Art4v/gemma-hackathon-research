"""ConcatAvgMaxPooling: cat(AvgPool2d(k, stride=1)(x), MaxPool2d(k, stride=1)(x), dim=1)."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'ConcatAvgMaxPooling_5794',
    'batch': 16,
    'channels': 64,
    'height': 256,
    'width': 256,
    'kernel_size': 12,
    'stride': 1,
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    x = jax.random.uniform(
        key,
        (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width']),
        dtype,
    )
    return (x,)


def workload(x):
    k = CONFIG['kernel_size']
    s = CONFIG['stride']
    window = (1, 1, k, k)
    strides = (1, 1, s, s)
    # No padding, so every window is full and the average is sum / (k*k).
    summed = lax.reduce_window(x, jnp.array(0, x.dtype), lax.add, window, strides, 'VALID')
    avg = summed / jnp.array(k * k, x.dtype)
    mx = lax.reduce_window(x, jnp.array(-jnp.inf, x.dtype), lax.max, window, strides, 'VALID')
    return jnp.concatenate([avg, mx], axis=1)
