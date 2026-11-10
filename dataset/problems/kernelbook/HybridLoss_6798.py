"""[bf16 variant] HybridLoss: flatten -> binary Dice loss + binary cross-entropy (log clamped at -100), batch-mean reduced."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'HybridLoss_6798',
    'batch': 8,
    'channels': 128,
    'height': 128,
    'width': 128,
    'smooth': 1.0,
    'exponent': 1,
    'balance_weight': 1.0,
    'dice_loss_weight': 1.0,
    'bce_loss_weight': 1.0,
    'bce_log_min': -100.0,
}


def create_inputs(dtype=jnp.bfloat16):
    shape = (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width'])
    key = jax.random.key(0)
    k_pred, k_target = jax.random.split(key)
    pred = jax.random.uniform(k_pred, shape, dtype)
    target = jax.random.uniform(k_target, shape, dtype)
    return pred, target


def workload(pred, target):
    pred = pred.reshape(pred.shape[0], -1)
    target = target.reshape(target.shape[0], -1)

    # Binary Dice loss
    smooth = CONFIG['smooth']
    exponent = CONFIG['exponent']
    num = jnp.sum(pred * target, axis=1) * 2 + smooth
    den = (
        CONFIG['balance_weight'] * jnp.sum(pred ** exponent, axis=1)
        + jnp.sum(target ** exponent, axis=1)
        + smooth
    )
    dice = jnp.mean(1 - num / den, axis=0)

    # Binary cross-entropy with PyTorch's log clamp
    log_min = CONFIG['bce_log_min']
    log_p = jnp.maximum(jnp.log(pred), log_min)
    log_1mp = jnp.maximum(jnp.log(1 - pred), log_min)
    bce = (target - 1) * log_1mp - target * log_p
    bce = jnp.mean(jnp.mean(bce, axis=1), axis=0)

    return CONFIG['dice_loss_weight'] * dice + CONFIG['bce_loss_weight'] * bce
