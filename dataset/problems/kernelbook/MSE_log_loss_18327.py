"""MSE log loss: clamp(pred, 0) -> log(pred+eps) - log(gt+eps) -> masked (gt > 0) mean of squared error."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'MSE_log_loss_18327',
    'batch': 8,
    'channels': 16,
    'height': 512,
    'width': 512,
    'eps': 1e-6,
    'clamp_min': 0.0,
}


def create_inputs(dtype=jnp.float32):
    shape = (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width'])
    k_pred, k_gt = jax.random.split(jax.random.key(0))
    prediction = jax.random.uniform(k_pred, shape, dtype)
    gt = jax.random.uniform(k_gt, shape, dtype)
    return prediction, gt


def workload(prediction, gt):
    eps = CONFIG['eps']
    pred = jnp.maximum(prediction, CONFIG['clamp_min'])
    err = jnp.log(pred + eps) - jnp.log(gt + eps)
    mask = gt > 0
    # Static-shape version of torch.mean(err[mask] ** 2); where() keeps masked-out NaNs out of the sum.
    sq_sum = jnp.sum(jnp.where(mask, err * err, 0.0))
    count = jnp.sum(mask).astype(err.dtype)
    return sq_sum / count
