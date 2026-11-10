"""Soft Dice loss: elementwise product, sums over spatial axes, smoothed ratio, mean, one minus."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'DiceLoss_1397',
    'batch': 8,
    'channels': 4,
    'depth': 128,
    'height': 128,
    'width': 128,
    'axis': (2, 3, 4),
    'smooth': 1e-07,
}


def create_inputs(dtype=jnp.float32):
    shape = (
        CONFIG['batch'],
        CONFIG['channels'],
        CONFIG['depth'],
        CONFIG['height'],
        CONFIG['width'],
    )
    key = jax.random.key(0)
    k_input, k_target = jax.random.split(key)
    input_ = jax.random.uniform(k_input, shape, dtype)
    target = jax.random.uniform(k_target, shape, dtype)
    return input_, target


def workload(input_, target):
    axis = CONFIG['axis']
    smooth = CONFIG['smooth']
    numerator = jnp.sum(input_ * target, axis=axis)
    denominator = jnp.sum(input_, axis=axis) + jnp.sum(target, axis=axis) + smooth
    dice = 2.0 * (numerator + smooth) / denominator
    return 1.0 - jnp.mean(dice)
