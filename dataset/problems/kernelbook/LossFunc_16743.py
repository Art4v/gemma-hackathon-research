"""[bf16 variant] LossFunc_16743: L1-gradient + MSE basic loss and per-frame annealed loss (pad, difference, crop, abs, mean, alpha**step scale)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'LossFunc_16743',
    'batch': 8,
    'num_frames': 8,
    'channels': 8,
    'height': 512,
    'width': 512,
    'coeff_basic': 1.0,
    'coeff_anneal': 1.0,
    'alpha': 0.9998,
    'beta': 100.0,
}


def create_inputs(dtype=jnp.bfloat16):
    k_pred_i, k_pred, k_gt, k_step = jax.random.split(jax.random.key(0), 4)
    b, n = CONFIG['batch'], CONFIG['num_frames']
    c, h, w = CONFIG['channels'], CONFIG['height'], CONFIG['width']
    pred_img_i = jax.random.uniform(k_pred_i, (b, n, h, w), dtype)
    pred_img = jax.random.uniform(k_pred, (b, c, h, w), dtype)
    ground_truth = jax.random.uniform(k_gt, (b, c, h, w), dtype)
    global_step = jax.random.uniform(k_step, (b, c, h, w), dtype)
    return pred_img_i, pred_img, ground_truth, global_step


def _tensor_gradient(img):
    h, w = img.shape[-2], img.shape[-1]
    lead_w = [(0, 0)] * (img.ndim - 1)
    lead_h = [(0, 0)] * (img.ndim - 2)
    l = jnp.pad(img, lead_w + [(1, 0)])
    r = jnp.pad(img, lead_w + [(0, 1)])
    u = jnp.pad(img, lead_h + [(1, 0), (0, 0)])
    d = jnp.pad(img, lead_h + [(0, 1), (0, 0)])
    return jnp.abs((l - r)[..., 0:h, 0:w]) + jnp.abs((u - d)[..., 0:h, 0:w])


def _loss_basic(pred, ground_truth):
    mse = jnp.mean((pred - ground_truth) ** 2)
    l1 = jnp.mean(jnp.abs(_tensor_gradient(pred) - _tensor_gradient(ground_truth)))
    return mse + l1


def workload(pred_img_i, pred_img, ground_truth, global_step):
    num_frames = pred_img_i.shape[1]
    anneal = 0.0
    for i in range(num_frames):
        anneal = anneal + _loss_basic(pred_img_i[:, i, ...], ground_truth)
    anneal = anneal / num_frames
    anneal = CONFIG['beta'] * jnp.power(CONFIG['alpha'], global_step) * anneal
    basic = _loss_basic(pred_img, ground_truth)
    return CONFIG['coeff_basic'] * basic, CONFIG['coeff_anneal'] * anneal
