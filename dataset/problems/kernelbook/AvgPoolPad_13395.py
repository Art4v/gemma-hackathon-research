"""[bf16 variant] ZeroPad2d((1,0,1,0)) -> AvgPool2d(3, stride=2, padding=1, count_include_pad=False) -> crop [1:, 1:]."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'AvgPoolPad_13395',
    'batch': 16,
    'channels': 64,
    'height': 256,
    'width': 256,
    'zero_pad': (1, 0, 1, 0),  # (left, right, top, bottom), like nn.ZeroPad2d
    'kernel_size': 3,
    'stride': 2,
    'padding': 1,
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    x = jax.random.uniform(
        key, (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width']), dtype)
    return (x,)


def workload(x):
    left, right, top, bottom = CONFIG['zero_pad']
    k, s, p = CONFIG['kernel_size'], CONFIG['stride'], CONFIG['padding']
    # ZeroPad2d zeros are real values and count toward the average.
    xp = jnp.pad(x, ((0, 0), (0, 0), (top, bottom), (left, right)))
    window = (1, 1, k, k)
    strides = (1, 1, s, s)
    pads = ((0, 0), (0, 0), (p, p), (p, p))
    sums = lax.reduce_window(xp, jnp.zeros((), x.dtype), lax.add, window, strides, pads)
    # count_include_pad=False leaves only the pool's own padding out of the divisor.
    ones = jnp.ones((1, 1) + xp.shape[2:], x.dtype)
    counts = lax.reduce_window(ones, jnp.zeros((), x.dtype), lax.add, window, strides, pads)
    out = sums / counts
    return out[:, :, 1:, 1:]
