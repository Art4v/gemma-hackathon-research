"""[bf16 variant] BCEWithLogits + per-channel sigmoid Dice loss, alpha*BCE + beta*(1 - mean dice)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'BCEDiceLoss_4907',
    'batch': 16,
    'channels': 8,
    'height': 512,
    'width': 512,
    'alpha': 4.0,
    'beta': 4.0,
    'dice_eps': 1e-6,
}


def create_inputs(dtype=jnp.bfloat16):
    k1, k2 = jax.random.split(jax.random.key(0))
    shape = (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width'])
    x = jax.random.uniform(k1, shape, dtype)
    target = jax.random.uniform(k2, shape, dtype)
    return x, target


def workload(x, target):
    # Stable BCE-with-logits, mean reduction (PyTorch default).
    bce = jnp.mean(jnp.maximum(x, 0) - x * target + jnp.log1p(jnp.exp(-jnp.abs(x))))

    # Per-channel Dice: reduce over every axis except channel (axis 1).
    p = jax.nn.sigmoid(x)
    axes = (0,) + tuple(range(2, x.ndim))
    intersect = jnp.sum(p * target, axis=axes)
    denom = jnp.sum(p * p, axis=axes) + jnp.sum(target * target, axis=axes)
    dice = 2.0 * (intersect / jnp.maximum(denom, CONFIG['dice_eps']))
    dice_loss = 1.0 - jnp.mean(dice)

    return CONFIG['alpha'] * bce + CONFIG['beta'] * dice_loss
