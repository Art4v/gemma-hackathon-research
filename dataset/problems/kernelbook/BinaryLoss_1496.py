"""BinaryLoss: pairwise L2 distance (query-pos, query-neg) -> squared pos distance + squared hinge on neg distance."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'BinaryLoss_1496',
    'batch': 16,
    'channels': 64,
    'height': 128,
    'width': 128,
    'margin': 1.0,
    'eps': 1e-6,
}


def create_inputs(dtype=jnp.float32):
    shape = (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width'])
    kq, kp, kn = jax.random.split(jax.random.key(0), 3)
    query = jax.random.uniform(kq, shape, dtype)
    positive = jax.random.uniform(kp, shape, dtype)
    negative = jax.random.uniform(kn, shape, dtype)
    return query, positive, negative


def _pairwise_distance(x1, x2):
    # Match torch: eps is added to the difference before the norm.
    diff = x1 - x2 + CONFIG['eps']
    return jnp.sqrt(jnp.sum(diff * diff, axis=-1))


def workload(query, positive, negative):
    d_pos = _pairwise_distance(query, positive)
    d_neg = _pairwise_distance(query, negative)
    hinge = jnp.maximum(CONFIG['margin'] - d_neg, 0.0)
    return d_pos ** 2 + hinge ** 2
