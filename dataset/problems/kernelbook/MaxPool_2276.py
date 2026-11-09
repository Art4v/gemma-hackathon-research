"""MaxPool2d(kernel=4, stride=1, padding=1) with optional ZeroPad2d((1,0,1,0)) + crop."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'MaxPool_2276',
    'batch': 16,
    'channels': 64,
    'height': 256,
    'width': 256,
    'kernel_size': 4,
    'stride': 1,
    'padding': 1,
    'zero_pad': False,
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    x = jax.random.uniform(
        key, (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width']), dtype)
    return (x,)


def workload(x):
    k = CONFIG['kernel_size']
    s = CONFIG['stride']
    p = CONFIG['padding']
    if CONFIG['zero_pad']:
        x = jnp.pad(x, ((0, 0), (0, 0), (1, 0), (1, 0)))
    out = lax.reduce_window(
        x, -jnp.inf, lax.max,
        window_dimensions=(1, 1, k, k),
        window_strides=(1, 1, s, s),
        padding=((0, 0), (0, 0), (p, p), (p, p)),
    )
    if CONFIG['zero_pad']:
        out = out[:, :, 1:, 1:]
    return out
