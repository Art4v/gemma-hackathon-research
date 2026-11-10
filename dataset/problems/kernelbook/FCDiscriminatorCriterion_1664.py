"""Binary cross-entropy with logits (reduction='none') followed by a per-sample mean over non-batch axes."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'FCDiscriminatorCriterion_1664',
    'batch': 8,
    'channels': 128,
    'height': 256,
    'width': 256,
    'reduce_axes': (1, 2, 3),
}


def create_inputs(dtype=jnp.float32):
    shape = (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width'])
    k_pred, k_gt = jax.random.split(jax.random.key(0))
    pred = jax.random.uniform(k_pred, shape, dtype)
    gt = jax.random.uniform(k_gt, shape, dtype)
    return pred, gt


def workload(pred, gt):
    loss = (1 - gt) * pred + jnp.logaddexp(jnp.zeros_like(pred), -pred)
    return jnp.mean(loss, axis=CONFIG['reduce_axes'])
