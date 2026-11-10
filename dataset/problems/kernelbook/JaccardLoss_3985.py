"""Per-sample soft Jaccard loss: flatten, intersection and union sums, smoothed ratio, batch mean."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'JaccardLoss_3985',
    'batch': 8,
    'channels': 128,
    'height': 128,
    'width': 256,
    'smooth': 1.0,
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    k_pred, k_target = jax.random.split(key)
    shape = (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width'])
    pred = jax.random.uniform(k_pred, shape, dtype)
    target = jax.random.uniform(k_target, shape, dtype)
    return pred, target


def workload(pred, target):
    smooth = CONFIG['smooth']
    batch = pred.shape[0]
    iflat = pred.reshape(batch, -1)
    tflat = target.reshape(batch, -1)
    intersection = jnp.sum(iflat * tflat, axis=1)
    union = jnp.sum(iflat, axis=1) + jnp.sum(tflat, axis=1) - intersection
    losses = 1 - (intersection + smooth) / (union + smooth)
    return jnp.sum(losses) / float(batch)
