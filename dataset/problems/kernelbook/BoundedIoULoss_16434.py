"""BoundedIoULoss: box center/size deltas -> clamped ratio and min ratio -> concat -> smooth-L1 -> sum."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'BoundedIoULoss_16434',
    'batch': 8,
    'channels': 4,
    'height': 1024,
    'width': 1024,
    'split': 2,
    'beta': 0.2,
    'eps': 0.001,
    'beta_threshold': 1e-05,
}


def create_inputs(dtype=jnp.float32):
    shape = (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width'])
    key = jax.random.key(0)
    k_pred, k_target = jax.random.split(key)
    pred = jax.random.uniform(k_pred, shape, dtype)
    target = jax.random.uniform(k_target, shape, dtype)
    return pred, target


def workload(pred, target):
    split = CONFIG['split']
    beta = CONFIG['beta']
    eps = CONFIG['eps']

    pred_lo, pred_hi = pred[:, :split], pred[:, split:]
    target_lo, target_hi = target[:, :split], target[:, split:]

    pred_ctr_2x = pred_lo + pred_hi
    pred_wh = pred_hi - pred_lo
    target_ctr_2x = target_lo + target_hi
    target_wh = target_hi - target_lo

    d_xy_2x = jnp.abs(target_ctr_2x - pred_ctr_2x)
    loss_xy = jnp.maximum((target_wh - d_xy_2x) / (target_wh + d_xy_2x + eps), 0)
    loss_wh = jnp.minimum(target_wh / (pred_wh + eps), pred_wh / (target_wh + eps))
    loss = 1 - jnp.concatenate([loss_xy, loss_wh], axis=-1)

    if beta >= CONFIG['beta_threshold']:
        loss = jnp.where(loss < beta, 0.5 * loss ** 2 / beta, loss - 0.5 * beta)

    return jnp.sum(loss)
