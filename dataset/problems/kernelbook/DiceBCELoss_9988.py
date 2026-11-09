"""DiceBCELoss: per-sample Dice coefficient (smooth=1) averaged over batch, plus mean binary cross-entropy."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'DiceBCELoss_9988',
    'batch': 16,
    'channels': 8,
    'height': 512,
    'width': 512,
    'smooth': 1.0,
    'log_clamp': -100.0,
}


def create_inputs(dtype=jnp.float32):
    shape = (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width'])
    k1, k2 = jax.random.split(jax.random.key(0))
    predicted = jax.random.uniform(k1, shape, dtype)
    target = jax.random.uniform(k2, shape, dtype)
    return predicted, target


def workload(predicted, target):
    smooth = CONFIG['smooth']
    axes = tuple(range(1, predicted.ndim))
    intersection = jnp.sum(predicted * target, axis=axes)
    coefficient = (2.0 * intersection + smooth) / (
        jnp.sum(predicted, axis=axes) + jnp.sum(target, axis=axes) + smooth)
    dice = jnp.mean(coefficient)
    # PyTorch's binary_cross_entropy clamps each log term at -100.
    log_p = jnp.maximum(jnp.log(predicted), CONFIG['log_clamp'])
    log_1mp = jnp.maximum(jnp.log(1.0 - predicted), CONFIG['log_clamp'])
    bce = jnp.mean(-(target * log_p + (1.0 - target) * log_1mp))
    return bce + (1.0 - dice)
