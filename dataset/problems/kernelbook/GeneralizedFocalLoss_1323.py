"""Generalized focal loss: |t-p|^beta weighted BCE with per-(batch, class) weights, summed and divided by the positive count."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'GeneralizedFocalLoss_1323',
    'batch': 16,
    'channels': 128,
    'height': 128,
    'width': 128,
    'beta': 2,
    'pred_eps': 1e-6,
}


def create_inputs(dtype=jnp.float32):
    k1, k2, k3 = jax.random.split(jax.random.key(0), 3)
    shape = (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width'])
    eps = CONFIG['pred_eps']
    # Keep prediction strictly inside (0, 1) so log(p) and log(1-p) stay finite.
    prediction = jax.random.uniform(k1, shape, dtype, minval=eps, maxval=1.0 - eps)
    target = jax.random.uniform(k2, shape, dtype)
    cls_weights = jax.random.uniform(k3, (CONFIG['batch'], CONFIG['channels']), dtype)
    return prediction, target, cls_weights


def workload(prediction, target, cls_weights):
    w = cls_weights[:, :, None, None]
    num_positive = jnp.sum((target == 1).astype(prediction.dtype))
    scale = jnp.power(jnp.abs(target - prediction), CONFIG['beta'])
    bce = target * jnp.log(prediction) + (1 - target) * jnp.log(1 - prediction)
    loss = -scale * bce * w
    return jnp.sum(loss) / jnp.maximum(1.0, num_positive)
