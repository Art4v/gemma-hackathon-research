"""EstimationLoss: focal-style binary cross-entropy (mean over N,C, sum) + positive/negative dice loss."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'EstimationLoss_8171',
    'batch': 16,
    'channels': 128,
    'height': 128,
    'width': 128,
    'gamma': 0,
    'alpha': 0,  # unused by forward, kept to mirror the module
    'log_eps': 1e-6,
    'dice_eps': 1e-6,
    'dice_weight': 1.0,
}


def create_inputs(dtype=jnp.float32):
    shape = (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width'])
    k1, k2 = jax.random.split(jax.random.key(0))
    pred = jax.random.uniform(k1, shape, dtype)
    target = jax.random.uniform(k2, shape, dtype)
    return pred, target


def workload(pred, target):
    gamma = CONFIG['gamma']
    log_eps = CONFIG['log_eps']
    dice_eps = CONFIG['dice_eps']

    temp1 = -(pred ** gamma) * ((1 - target) * jnp.log(1 - pred + log_eps))
    temp2 = -((1 - pred) ** gamma) * (target * jnp.log(pred + log_eps))
    ce_loss = jnp.sum(jnp.mean(temp1 + temp2, axis=(0, 1)))

    abs_p = jnp.abs(pred)
    abs_t = jnp.abs(target)
    inter_pos = jnp.sum(pred * target, axis=1)
    card_pos = jnp.sum(abs_p + abs_t, axis=1)
    dice_pos = (inter_pos + dice_eps) / (card_pos + dice_eps)
    inter_neg = jnp.sum((1.0 - pred) * (1.0 - target), axis=1)
    card_neg = jnp.sum(2 - abs_p - abs_t, axis=1)
    dice_neg = (inter_neg + dice_eps) / (card_neg + dice_eps)
    dice_loss = jnp.sum(jnp.mean(1.5 - dice_pos - dice_neg, axis=0))

    return ce_loss + CONFIG['dice_weight'] * dice_loss
