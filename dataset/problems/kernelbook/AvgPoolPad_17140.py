"""[bf16 variant] ZeroPad2d((1,0,1,0)) -> AvgPool2d(3, stride=2, padding=1, count_include_pad=False) -> slice [:, :, 1:, 1:]."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'AvgPoolPad_17140',
    'batch': 16,
    'channels': 64,
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
    # ZeroPad2d: these zeros are real elements and are counted by the pool.
    xp = jnp.pad(x, ((0, 0), (0, 0), (1, 0), (1, 0)))
    window = (1, 1, k, k)
    strides = (1, 1, s, s)
    pads = ((0, 0), (0, 0), (p, p), (p, p))
    sums = lax.reduce_window(xp, jnp.array(0, xp.dtype), lax.add, window, strides, pads)
    # count_include_pad=False: divide by number of in-bounds elements per window.
    ones = jnp.ones((1, 1) + xp.shape[2:], xp.dtype)
    counts = lax.reduce_window(ones, jnp.array(0, xp.dtype), lax.add, window, strides, pads)
    out = sums / counts
    return out[:, :, 1:, 1:]
