"""Dice_5301: softmax(dim=1) -> argmax -> per-channel dice vs center-cropped y_true -> mean."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'Dice_5301',
    'batch': 8,
    'channels': 4,
    'height': 1024,
    'width': 1024,
    'true_height': 1024,
    'true_width': 1024,
    'eps': 1e-15,
    'bback_ignore': True,
}


def create_inputs(dtype=jnp.float32):
    k1, k2 = jax.random.split(jax.random.key(0))
    n, c = CONFIG['batch'], CONFIG['channels']
    y_pred = jax.random.uniform(k1, (n, c, CONFIG['height'], CONFIG['width']), dtype)
    y_true = jax.random.uniform(k2, (n, c, CONFIG['true_height'], CONFIG['true_width']), dtype)
    return (y_pred, y_true)


def _centercrop(image, w, h):
    ht, wt = image.shape[2], image.shape[3]
    padw, padh = (wt - w) // 2, (ht - h) // 2
    if padw > 0 and padh > 0:
        image = image[:, :, padh:ht - padh, padw:wt - padw]
    return image


def workload(y_pred, y_true):
    _n, ch, h, w = y_pred.shape
    y_true = _centercrop(y_true, w, h)
    prob = jax.nn.softmax(y_pred, axis=1)
    pred = jnp.argmax(prob, axis=1).astype(jnp.float32)  # (N, H, W), class index as float
    start = int(CONFIG['bback_ignore'])
    yt = y_true[:, start:ch].astype(jnp.float32)  # (N, C', H, W)
    inter = jnp.sum(yt * pred[:, None], axis=(0, 2, 3))
    true_sum = jnp.sum(yt, axis=(0, 2, 3))
    pred_sum = jnp.sum(pred)
    dices = 2.0 * inter / (true_sum + pred_sum + CONFIG['eps']) * 100
    return jnp.mean(dices)
