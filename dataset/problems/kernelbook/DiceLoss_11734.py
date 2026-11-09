"""Dice loss: per-sample (2*sum(p*t)+smooth)/(sum(p)+sum(t)+smooth), then 1 - batch mean."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'DiceLoss_11734',
    'batch': 16,
    'channels': 4,
    'height': 512,
    'width': 512,
    'smooth': 1.0,
}


def create_inputs(dtype=jnp.float32):
    k1, k2 = jax.random.split(jax.random.key(0))
    shape = (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width'])
    predicted = jax.random.uniform(k1, shape, dtype)
    target = jax.random.uniform(k2, shape, dtype)
    return predicted, target


def workload(predicted, target):
    smooth = CONFIG['smooth']
    axes = tuple(range(1, predicted.ndim))
    intersection = jnp.sum(predicted * target, axis=axes)
    denom = jnp.sum(predicted, axis=axes) + jnp.sum(target, axis=axes) + smooth
    coefficient = (2.0 * intersection + smooth) / denom
    return 1.0 - jnp.mean(coefficient)
