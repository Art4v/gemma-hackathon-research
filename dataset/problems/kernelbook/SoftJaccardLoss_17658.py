"""SoftJaccardLoss: sigmoid -> per-sample soft intersection/union -> 1 - mean Jaccard."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'SoftJaccardLoss_17658',
    'batch': 32,
    'channels': 4,
    'height': 512,
    'width': 512,
    'smooth': 1e-15,
    'use_log': False,
}


def create_inputs(dtype=jnp.float32):
    k1, k2 = jax.random.split(jax.random.key(0))
    shape = (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width'])
    logits = jax.random.uniform(k1, shape, dtype)
    labels = jax.random.uniform(k2, shape, dtype)
    return logits, labels


def workload(logits, labels):
    num = labels.shape[0]
    eps = CONFIG['smooth']
    m1 = jax.nn.sigmoid(logits.reshape(num, -1))
    m2 = labels.reshape(num, -1)
    intersection = jnp.sum(m1 * m2, axis=1)
    score = (intersection + eps) / (jnp.sum(m1, axis=1) + jnp.sum(m2, axis=1) - intersection + eps)
    jaccard = jnp.sum(score, axis=0) / num
    if CONFIG['use_log']:
        return -jnp.log(jaccard)
    return 1 - jaccard
