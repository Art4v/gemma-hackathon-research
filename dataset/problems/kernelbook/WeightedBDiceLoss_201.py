"""Weighted binary Dice loss: centercrop(y_true, weight) -> sigmoid(y_pred) -> weighted Dice score -> 1 - score."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'WeightedBDiceLoss_201',
    'batch': 16,
    'channels': 4,
    'height': 1024,
    'width': 1024,
    'smooth': 1.0,
}


def create_inputs(dtype=jnp.float32):
    shape = (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width'])
    k1, k2, k3 = jax.random.split(jax.random.key(0), 3)
    y_pred = jax.random.uniform(k1, shape, dtype)
    y_true = jax.random.uniform(k2, shape, dtype)
    weight = jax.random.uniform(k3, shape, dtype)
    return y_pred, y_true, weight


def _centercrop(image, w, h):
    # Matches the PyTorch helper: crop only when both pads are positive.
    ht, wt = image.shape[2], image.shape[3]
    padw, padh = (wt - w) // 2, (ht - h) // 2
    if padw > 0 and padh > 0:
        image = image[:, :, padh:ht - padh, padw:wt - padw]
    return image


def workload(y_pred, y_true, weight):
    h, w = y_pred.shape[2], y_pred.shape[3]
    m1 = _centercrop(y_true, w, h)
    wt = _centercrop(weight, w, h)
    m2 = jax.nn.sigmoid(y_pred)
    smooth = CONFIG['smooth']
    score = (2.0 * jnp.sum(wt * m1 * m2) + smooth) / (
        jnp.sum(wt * m1) + jnp.sum(wt * m2) + smooth)
    return 1.0 - jnp.sum(score)
