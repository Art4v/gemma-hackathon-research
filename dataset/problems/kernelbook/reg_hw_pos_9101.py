"""SmoothL1 height/width regression ratios weighted by a label mask, summed and normalized by max(1, 2*mask sum)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'reg_hw_pos_9101',
    'batch': 16,
    'channels': 4,
    'height': 512,
    'width': 512,
    'eps': 1e-10,
    'beta': 1.0,
    'h_channel': 0,
    'w_channel': 1,
    'mask_channel': 2,
    'min_denominator': 1.0,
    'denominator_scale': 2,
}


def create_inputs(dtype=jnp.float32):
    shape = (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width'])
    key = jax.random.key(0)
    k_pred, k_label = jax.random.split(key)
    h_pred = jax.random.uniform(k_pred, shape, dtype)
    h_label = jax.random.uniform(k_label, shape, dtype)
    return h_pred, h_label


def _smooth_l1(pred, target):
    beta = CONFIG['beta']
    diff = jnp.abs(pred - target)
    return jnp.where(diff < beta, 0.5 * diff * diff / beta, diff - 0.5 * beta)


def workload(h_pred, h_label):
    eps = CONFIG['eps']
    mask = h_label[:, CONFIG['mask_channel'], :, :]

    label_h = h_label[:, CONFIG['h_channel'], :, :]
    pred_h = h_pred[:, CONFIG['h_channel'], :, :]
    denom_h = label_h + eps
    l1_loss = mask * _smooth_l1(pred_h / denom_h, label_h / denom_h)

    label_w = h_label[:, CONFIG['w_channel'], :, :]
    pred_w = h_pred[:, CONFIG['w_channel'], :, :]
    denom_w = label_w + eps
    l1_loss = l1_loss + mask * _smooth_l1(pred_w / denom_w, label_w / denom_w)

    min_denom = CONFIG['min_denominator']
    scaled_mask_sum = jnp.sum(mask) * CONFIG['denominator_scale']
    denominator = jnp.where(scaled_mask_sum > min_denom, scaled_mask_sum, min_denom)
    return jnp.sum(l1_loss) / denominator
