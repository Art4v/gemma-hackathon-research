"""[bf16 variant] BCE-with-logits (mean) + soft Dice loss: 0.5 * bce + (1 - mean_dice)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'BCEDiceLoss_5703',
    'batch': 16,
    'channels': 4,
    'height': 512,
    'width': 512,
    'smooth': 1e-05,
    'bce_weight': 0.5,
}


def create_inputs(dtype=jnp.bfloat16):
    k1, k2 = jax.random.split(jax.random.key(0))
    shape = (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width'])
    logits = jax.random.uniform(k1, shape, dtype)
    target = jax.random.uniform(k2, shape, dtype)
    return logits, target


def workload(logits, target):
    smooth = CONFIG['smooth']
    # Stable form used by PyTorch's binary_cross_entropy_with_logits.
    bce = jnp.mean(jnp.maximum(logits, 0) - logits * target + jnp.log1p(jnp.exp(-jnp.abs(logits))))
    num = target.shape[0]
    p = jax.nn.sigmoid(logits).reshape(num, -1)
    t = target.reshape(num, -1)
    dice = (2.0 * jnp.sum(p * t, axis=1) + smooth) / (jnp.sum(p, axis=1) + jnp.sum(t, axis=1) + smooth)
    dice_loss = 1.0 - jnp.sum(dice) / num
    return CONFIG['bce_weight'] * bce + dice_loss
