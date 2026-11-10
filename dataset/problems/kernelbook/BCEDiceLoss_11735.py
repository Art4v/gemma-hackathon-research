"""[bf16 variant] BCEDiceLoss: binary_cross_entropy(mean, log clamped at -100) + (1 - mean per-sample Dice coefficient)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'BCEDiceLoss_11735',
    'batch': 16,
    'channels': 8,
    'height': 512,
    'width': 512,
    'smooth': 1.0,
    'log_clamp': -100.0,
}


def create_inputs(dtype=jnp.bfloat16):
    k1, k2 = jax.random.split(jax.random.key(0))
    shape = (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width'])
    predicted = jax.random.uniform(k1, shape, dtype)
    target = jax.random.uniform(k2, shape, dtype)
    return predicted, target


def workload(predicted, target):
    clamp = CONFIG['log_clamp']
    smooth = CONFIG['smooth']
    # PyTorch clamps each log term at -100 so p == 0 or p == 1 stays finite.
    log_p = jnp.maximum(jnp.log(predicted), clamp)
    log_1mp = jnp.maximum(jnp.log1p(-predicted), clamp)
    bce = jnp.mean(-(target * log_p + (1.0 - target) * log_1mp))

    axes = tuple(range(1, predicted.ndim))
    intersection = jnp.sum(predicted * target, axis=axes)
    denom = jnp.sum(predicted, axis=axes) + jnp.sum(target, axis=axes) + smooth
    coefficient = (2.0 * intersection + smooth) / denom
    dice = 1.0 - jnp.mean(coefficient)
    return bce + dice
