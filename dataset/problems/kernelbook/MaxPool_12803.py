"""[bf16 variant] MaxPool2d(kernel=4, stride=1, padding=1) with optional ZeroPad2d((1,0,1,0)) + crop."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'MaxPool_12803',
    'batch': 8,
    'channels': 128,
    'height': 256,
    'width': 256,
    'kernel_size': 4,
    'stride': 1,
    'padding': 1,
    'zero_pad': False,
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    x = jax.random.uniform(
        key, (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width']), dtype)
    return (x,)


def workload(x):
    k, s, p = CONFIG['kernel_size'], CONFIG['stride'], CONFIG['padding']
    if CONFIG['zero_pad']:
        x = jnp.pad(x, ((0, 0), (0, 0), (1, 0), (1, 0)))
    # PyTorch max-pool padding behaves like -inf (padded cells never win).
    y = lax.reduce_window(
        x, jnp.array(-jnp.inf, x.dtype), lax.max,
        window_dimensions=(1, 1, k, k),
        window_strides=(1, 1, s, s),
        padding=((0, 0), (0, 0), (p, p), (p, p)),
    )
    if CONFIG['zero_pad']:
        y = y[:, :, 1:, 1:]
    return y
