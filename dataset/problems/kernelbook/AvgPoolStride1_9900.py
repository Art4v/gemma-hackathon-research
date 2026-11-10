"""[bf16 variant] Replicate pad (right/bottom by 1) followed by 2x2 average pool with stride 1 on NCHW input."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'AvgPoolStride1_9900',
    'batch': 8,
    'channels': 128,
    'height': 256,
    'width': 256,
    'kernel_size': 2,
    'stride': 1,
    'pad_bottom': 1,
    'pad_right': 1,
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    shape = (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width'])
    x = jax.random.uniform(key, shape, dtype)
    return (x,)


def workload(x):
    k = CONFIG['kernel_size']
    s = CONFIG['stride']
    y = jnp.transpose(x, (0, 2, 3, 1))
    y = jnp.pad(
        y,
        ((0, 0), (0, CONFIG['pad_bottom']), (0, CONFIG['pad_right']), (0, 0)),
        mode='edge',
    )
    y = lax.reduce_window(
        y,
        0.0,
        lax.add,
        window_dimensions=(1, k, k, 1),
        window_strides=(1, s, s, 1),
        padding='VALID',
    )
    y = y / (k * k)
    return jnp.transpose(y, (0, 3, 1, 2))
