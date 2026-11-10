"""TripletLoss: pairwise L2 distances (eps-shifted, last axis) -> hinge with margin -> mean."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'TripletLoss_4328',
    'batch': 8,
    'height': 128,
    'width': 128,
    'embed_dim': 128,
    'margin': 5.0,
    'p': 2,
    'eps': 1e-6,
}


def create_inputs(dtype=jnp.float32):
    shape = (CONFIG['batch'], CONFIG['height'], CONFIG['width'], CONFIG['embed_dim'])
    k_anchor, k_negative, k_positive = jax.random.split(jax.random.key(0), 3)
    anchor = jax.random.uniform(k_anchor, shape, dtype)
    negative = jax.random.uniform(k_negative, shape, dtype)
    positive = jax.random.uniform(k_positive, shape, dtype)
    return anchor, negative, positive


def workload(anchor, negative, positive):
    eps = CONFIG['eps']
    p = CONFIG['p']
    pos_dist = jnp.linalg.norm(anchor - positive + eps, ord=p, axis=-1)
    neg_dist = jnp.linalg.norm(anchor - negative + eps, ord=p, axis=-1)
    hinge_dist = jnp.maximum(CONFIG['margin'] + pos_dist - neg_dist, 0.0)
    return jnp.mean(hinge_dist)
