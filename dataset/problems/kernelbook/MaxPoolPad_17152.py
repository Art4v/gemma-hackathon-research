"""ZeroPad2d((1,0,1,0)) -> MaxPool2d(3, stride=2, padding=1) -> slice [:, :, 1:, 1:]."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'MaxPoolPad_17152',
    'batch': 16,
    'channels': 64,
    'height': 256,
    'width': 256,
    'zero_pad_top': 1,
    'zero_pad_left': 1,
    'pool_kernel': 3,
    'pool_stride': 2,
    'pool_padding': 1,
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
    top = CONFIG['zero_pad_top']
    left = CONFIG['zero_pad_left']
    k = CONFIG['pool_kernel']
    s = CONFIG['pool_stride']
    p = CONFIG['pool_padding']
    x = jnp.pad(x, ((0, 0), (0, 0), (top, 0), (left, 0)))
    y = lax.reduce_window(
        x,
        jnp.array(-jnp.inf, dtype=x.dtype),
        lax.max,
        window_dimensions=(1, 1, k, k),
        window_strides=(1, 1, s, s),
        padding=((0, 0), (0, 0), (p, p), (p, p)),
    )
    return y[:, :, top:, left:]
