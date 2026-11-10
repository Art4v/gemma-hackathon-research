"""TripletMarginLoss: pairwise L2 distance (eps added to difference) over last axis -> hinge with margin -> mean."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'TripletLoss_2156',
    'batch': 32,
    'channels': 128,
    'height': 128,
    'width': 128,
    'margin': 4.0,
    'p': 2,
    'eps': 1e-6,
}


def create_inputs(dtype=jnp.float32):
    shape = (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width'])
    key = jax.random.key(0)
    k_anchor, k_positive, k_negative = jax.random.split(key, 3)
    anchor = jax.random.uniform(k_anchor, shape, dtype)
    positive = jax.random.uniform(k_positive, shape, dtype)
    negative = jax.random.uniform(k_negative, shape, dtype)
    return anchor, positive, negative


def _pairwise_distance(x1, x2):
    return jnp.linalg.norm(x1 - x2 + CONFIG['eps'], ord=CONFIG['p'], axis=-1)


def workload(anchor, positive, negative):
    dist_pos = _pairwise_distance(anchor, positive)
    dist_neg = _pairwise_distance(anchor, negative)
    losses = jnp.maximum(CONFIG['margin'] + dist_pos - dist_neg, 0.0)
    return jnp.mean(losses)
