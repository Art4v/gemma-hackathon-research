"""[bf16 variant] Replicate pad (right/bottom by 1) followed by 2x2 max pool with stride 1."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'MaxPoolStride1_7641',
    'batch': 8,
    'channels': 128,
    'height': 256,
    'width': 256,
    'pad_left': 0,
    'pad_right': 1,
    'pad_top': 0,
    'pad_bottom': 1,
    'kernel_size': 2,
    'stride': 1,
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
    k = CONFIG['kernel_size']
    s = CONFIG['stride']
    y = jnp.transpose(x, (0, 2, 3, 1))
    y = jnp.pad(
        y,
        (
            (0, 0),
            (CONFIG['pad_top'], CONFIG['pad_bottom']),
            (CONFIG['pad_left'], CONFIG['pad_right']),
            (0, 0),
        ),
        mode='edge',
    )
    y = lax.reduce_window(y, -jnp.inf, lax.max, (1, k, k, 1), (1, s, s, 1), 'VALID')
    return jnp.transpose(y, (0, 3, 1, 2))
