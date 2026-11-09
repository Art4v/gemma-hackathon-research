"""BCEWithLogits (mean) + sigmoid per-channel Dice loss, weighted sum."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'BCEDiceLoss_1717',
    'batch': 16,
    'channels': 128,
    'height': 128,
    'width': 128,
    'alpha': 4.0,
    'beta': 4.0,
    'eps': 1e-6,
}


def create_inputs(dtype=jnp.float32):
    k1, k2 = jax.random.split(jax.random.key(0))
    shape = (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width'])
    x = jax.random.uniform(k1, shape, dtype)
    target = jax.random.uniform(k2, shape, dtype)
    return x, target


def workload(x, target):
    # BCEWithLogitsLoss, mean reduction, stable form.
    bce = jnp.mean(jnp.maximum(x, 0) - x * target + jnp.log1p(jnp.exp(-jnp.abs(x))))
    # Per-channel Dice on sigmoid probabilities; reduce over all axes except channel.
    p = jax.nn.sigmoid(x)
    axes = (0,) + tuple(range(2, x.ndim))
    intersect = jnp.sum(p * target, axis=axes)
    denom = jnp.sum(p * p, axis=axes) + jnp.sum(target * target, axis=axes)
    dice = 2.0 * (intersect / jnp.maximum(denom, CONFIG['eps']))
    dice_loss = 1.0 - jnp.mean(dice)
    return CONFIG['alpha'] * bce + CONFIG['beta'] * dice_loss
