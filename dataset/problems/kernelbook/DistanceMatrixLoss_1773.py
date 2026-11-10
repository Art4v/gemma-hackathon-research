"""DistanceMatrixLoss: per-sample masked L1 sum divided by per-sample mask sum, then batch mean."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'DistanceMatrixLoss_1773',
    'batch': 8,
    'channels': 128,
    'height': 128,
    'width': 128,
}


def create_inputs(dtype=jnp.float32):
    shape = (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width'])
    k_scores, k_labels, k_mask = jax.random.split(jax.random.key(0), 3)
    scores = jax.random.uniform(k_scores, shape, dtype)
    labels = jax.random.uniform(k_labels, shape, dtype)
    mask = jax.random.uniform(k_mask, shape, dtype)
    return scores, labels, mask


def workload(scores, labels, mask):
    sq_lengths = mask.reshape(mask.shape[0], -1).sum(axis=1)
    l1_diff = (mask * jnp.abs(scores - labels)).reshape(labels.shape[0], -1).sum(axis=1)
    return jnp.mean(l1_diff / sq_lengths)
