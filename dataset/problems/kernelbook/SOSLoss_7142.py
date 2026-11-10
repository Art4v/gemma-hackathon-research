"""SOSLoss: sum over dim 1 of (a-n)^2 and (p-n)^2, then sqrt(sum((d_an - d_pn)^2)) / N."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'SOSLoss_7142',
    'batch': 16,
    'channels': 128,
    'height': 128,
    'width': 128,
}


def create_inputs(dtype=jnp.float32):
    shape = (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width'])
    ka, kp, kn = jax.random.split(jax.random.key(0), 3)
    anchors = jax.random.uniform(ka, shape, dtype)
    positives = jax.random.uniform(kp, shape, dtype)
    negatives = jax.random.uniform(kn, shape, dtype)
    return anchors, positives, negatives


def workload(anchors, positives, negatives):
    dist_an = jnp.sum(jnp.square(anchors - negatives), axis=1)
    dist_pn = jnp.sum(jnp.square(positives - negatives), axis=1)
    nq = anchors.shape[0]
    return jnp.sqrt(jnp.sum(jnp.square(dist_an - dist_pn))) / nq
