"""[bf16 variant] Zero pad (top/left 1) -> AvgPool2d(3, stride 2, pad 1, count_include_pad=False) -> slice [1:, 1:]."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'AvgPoolPad_5575',
    'batch': 8,
    'channels': 128,
    'height': 256,
    'width': 256,
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
    k, s, p = CONFIG['kernel_size'], CONFIG['stride'], CONFIG['padding']
    # ZeroPad2d((1, 0, 1, 0)): these zeros are real data and count toward the divisor.
    xp = jnp.pad(x, ((0, 0), (0, 0), (1, 0), (1, 0)))
    window = (1, 1, k, k)
    strides = (1, 1, s, s)
    pool_pad = ((0, 0), (0, 0), (p, p), (p, p))
    sums = lax.reduce_window(xp, jnp.array(0, xp.dtype), lax.add, window, strides, pool_pad)
    # count_include_pad=False: divisor counts only elements inside xp, not the pool padding.
    ones = jnp.ones((1, 1) + xp.shape[2:], xp.dtype)
    counts = lax.reduce_window(ones, jnp.array(0, xp.dtype), lax.add, window, strides, pool_pad)
    out = sums / counts
    return out[:, :, 1:, 1:]
