"""ZeroPad2d(1,0,1,0) -> AvgPool2d(3, stride 2, pad 1, count_include_pad=False) -> crop [1:, 1:]."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'AvgPoolPad_9866',
    'batch': 8,
    'channels': 128,
    'height': 256,
    'width': 256,
    'zero_pad': (1, 0, 1, 0),  # (left, right, top, bottom), as nn.ZeroPad2d
    'kernel_size': 3,
    'stride': 2,
    'padding': 1,
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    x = jax.random.uniform(
        key, (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width']), dtype)
    return (x,)


def workload(x):
    left, right, top, bottom = CONFIG['zero_pad']
    k, s, p = CONFIG['kernel_size'], CONFIG['stride'], CONFIG['padding']
    # Explicit zero pad: these zeros are real values and count in the average.
    xp = jnp.pad(x, ((0, 0), (0, 0), (top, bottom), (left, right)))
    window = (1, 1, k, k)
    strides = (1, 1, s, s)
    pads = ((0, 0), (0, 0), (p, p), (p, p))
    sums = lax.reduce_window(xp, jnp.array(0, xp.dtype), lax.add, window, strides, pads)
    # count_include_pad=False: divide by the number of non-pool-padding elements.
    ones = jnp.ones((1, 1) + xp.shape[2:], xp.dtype)
    counts = lax.reduce_window(ones, jnp.array(0, xp.dtype), lax.add, window, strides, pads)
    out = sums / counts
    return out[:, :, 1:, 1:]
