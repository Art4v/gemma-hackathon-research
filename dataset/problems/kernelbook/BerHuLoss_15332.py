"""[bf16 variant] BerHu (reverse Huber) loss: gt > eps validity mask -> abs diff -> scale * max threshold -> quadratic above threshold -> mean over valid pixels."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'BerHuLoss_15332',
    'batch': 8,
    'channels': 128,
    'height': 128,
    'width': 256,
    'scale': 0.5,
    'eps': 1e-05,
}


def create_inputs(dtype=jnp.bfloat16):
    shape = (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width'])
    key = jax.random.key(0)
    k_pred, k_gt = jax.random.split(key)
    pred = jax.random.uniform(k_pred, shape, dtype)
    gt = jax.random.uniform(k_gt, shape, dtype)
    return pred, gt


def workload(pred, gt):
    eps = CONFIG['eps']
    valid = gt > eps
    diff = jnp.where(valid, jnp.abs(pred - gt), jnp.zeros_like(pred))
    threshold = CONFIG['scale'] * lax.stop_gradient(jnp.max(diff))
    mask = diff > threshold
    berhu = (diff ** 2 + threshold ** 2) / (2 * threshold + eps)
    loss = jnp.where(mask, berhu, diff)
    return jnp.sum(loss) / jnp.sum(valid).astype(pred.dtype)
