"""ZeroPad2d((1,0,1,0)) -> MaxPool2d(3, stride=2, padding=1) -> crop [:, :, 1:, 1:]."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'MaxPoolPad_5576',
    'batch': 8,
    'channels': 128,
    'height': 256,
    'width': 256,
    'zero_pad': (1, 0, 1, 0),  # (left, right, top, bottom), like nn.ZeroPad2d
    'pool_kernel': 3,
    'pool_stride': 2,
    'pool_padding': 1,
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    x = jax.random.uniform(
        key, (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width']), dtype)
    return (x,)


def workload(x):
    left, right, top, bottom = CONFIG['zero_pad']
    x = jnp.pad(x, ((0, 0), (0, 0), (top, bottom), (left, right)))
    k, s, p = CONFIG['pool_kernel'], CONFIG['pool_stride'], CONFIG['pool_padding']
    # Max-pool padding is -inf (PyTorch semantics); the zero pad above is real data.
    x = lax.reduce_window(
        x, -jnp.inf, lax.max,
        window_dimensions=(1, 1, k, k),
        window_strides=(1, 1, s, s),
        padding=((0, 0), (0, 0), (p, p), (p, p)),
    )
    return x[:, :, 1:, 1:]
