"""[bf16 variant] ZeroPad2d((1,0,1,0)) -> AvgPool2d(3, stride=2, padding=1, count_include_pad=False) -> slice [:, :, 1:, 1:]."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'AvgPoolPad_17158',
    'batch': 8,
    'channels': 128,
    'height': 256,
    'width': 256,
    'kernel_size': 3,
    'stride': 2,
    'padding': 1,
    'zero_pad': (1, 0, 1, 0),  # (left, right, top, bottom), as nn.ZeroPad2d
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    x = jax.random.uniform(
        key,
        (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width']),
        dtype,
    )
    return (x,)


def workload(x):
    left, right, top, bottom = CONFIG['zero_pad']
    k, s, p = CONFIG['kernel_size'], CONFIG['stride'], CONFIG['padding']
    x = jnp.pad(x, ((0, 0), (0, 0), (top, bottom), (left, right)))
    window = (1, 1, k, k)
    strides = (1, 1, s, s)
    pool_pad = ((0, 0), (0, 0), (p, p), (p, p))
    sums = lax.reduce_window(x, jnp.array(0, x.dtype), lax.add, window, strides, pool_pad)
    # count_include_pad=False: the divisor counts only in-bounds elements of the
    # zero-padded tensor (ZeroPad2d zeros count, the pool's own padding does not).
    ones = jnp.ones((1, 1) + x.shape[2:], x.dtype)
    counts = lax.reduce_window(ones, jnp.array(0, x.dtype), lax.add, window, strides, pool_pad)
    out = sums / counts
    return out[:, :, 1:, 1:]
