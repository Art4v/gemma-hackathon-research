"""[bf16 variant] Replicate pad (F.pad mode='replicate') followed by MaxPool2d (kernel 4, stride 4, dilation 1)."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'PaddedMaxPool2d_2241',
    'batch': 16,
    'channels': 64,
    'height': 256,
    'width': 256,
    'kernel_size': 4,
    'stride': 4,
    'dilation': 1,
    # PyTorch F.pad order: (left, right, top, bottom)
    'padding': (0, 0, 0, 0),
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    x = jax.random.uniform(
        key, (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width']), dtype)
    return (x,)


def workload(x):
    left, right, top, bottom = CONFIG['padding']
    x = jnp.pad(x, ((0, 0), (0, 0), (top, bottom), (left, right)), mode='edge')
    k, s, d = CONFIG['kernel_size'], CONFIG['stride'], CONFIG['dilation']
    return lax.reduce_window(
        x, -jnp.inf, lax.max,
        window_dimensions=(1, 1, k, k),
        window_strides=(1, 1, s, s),
        padding='VALID',
        window_dilation=(1, 1, d, d),
    )
