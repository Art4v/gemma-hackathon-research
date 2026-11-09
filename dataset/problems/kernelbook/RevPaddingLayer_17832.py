"""[bf16 variant] AvgPool2d(k=3, pad=1, count_include_pad) -> concat zeros on both sides of the channel dim."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'RevPaddingLayer_17832',
    'batch': 8,
    'channels': 128,
    'height': 256,
    'width': 256,
    'kernel_size': 3,
    'stride': 1,
    'padding': 1,
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    x = jax.random.uniform(
        key, (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width']), dtype)
    return (x,)


def workload(x):
    k, s, p = CONFIG['kernel_size'], CONFIG['stride'], CONFIG['padding']
    summed = lax.reduce_window(
        x, jnp.array(0, x.dtype), lax.add,
        window_dimensions=(1, 1, k, k),
        window_strides=(1, 1, s, s),
        padding=((0, 0), (0, 0), (p, p), (p, p)),
    )
    # count_include_pad=True (PyTorch default): always divide by k*k.
    pooled = summed / (k * k)
    c = pooled.shape[1]
    c_left = (c + 1) // 2  # torch.chunk puts the ceil half first
    n, _, h, w = pooled.shape
    zeros_left = jnp.zeros((n, c_left, h, w), pooled.dtype)
    zeros_right = jnp.zeros((n, c - c_left, h, w), pooled.dtype)
    return jnp.concatenate([zeros_left, pooled, zeros_right], axis=1)
