"""SmoothL1Loss elementwise: |pred - target| -> where(diff < beta, 0.5*diff^2/beta, diff - 0.5*beta), no reduction."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'SmoothL1Loss_7106',
    'batch': 16,
    'channels': 32,
    'height': 256,
    'width': 256,
    'beta': 1.0,
}


def create_inputs(dtype=jnp.float32):
    shape = (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width'])
    k_pred, k_target = jax.random.split(jax.random.key(0))
    pred = jax.random.uniform(k_pred, shape, dtype)
    target = jax.random.uniform(k_target, shape, dtype)
    return pred, target


def workload(pred, target):
    beta = CONFIG['beta']
    diff = jnp.abs(pred - target)
    # ponytail: forward() never applies self.reduction, so the loss stays per-element.
    return jnp.where(diff < beta, 0.5 * diff * diff / beta, diff - 0.5 * beta)
