"""DisparityRegression: argmax over disparity axis -> clipped window gather -> normalize -> weighted sum of disparity indices."""

import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'DisparityRegression_1094',
    'batch': 8,
    'max_disp': 128,
    'height': 256,
    'width': 256,
    'win_size': 4,
    'eps': 1e-08,
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    x = jax.random.uniform(
        key,
        (CONFIG['batch'], CONFIG['max_disp'], CONFIG['height'], CONFIG['width']),
        dtype,
    )
    return (x,)


def workload(x):
    win_size = CONFIG['win_size']
    if win_size > 0:
        num_disp = x.shape[1]
        max_d = jnp.argmax(x, axis=1, keepdims=True)
        offsets = jnp.arange(-win_size, win_size + 1).reshape(1, -1, 1, 1)
        index = jnp.clip(max_d + offsets, 0, num_disp - 1)
        part_x = jnp.take_along_axis(x, index, axis=1)
        part_x = part_x / (jnp.sum(part_x, axis=1, keepdims=True) + CONFIG['eps'])
        part_d = index.astype(x.dtype)
        return jnp.sum(part_x * part_d, axis=1)
    disp = jnp.arange(0, CONFIG['max_disp'], dtype=x.dtype).reshape(1, -1, 1, 1)
    return jnp.sum(x * disp, axis=1)
