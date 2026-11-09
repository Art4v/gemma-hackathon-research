"""[m1 M3_ragged variant of kb/l1/99_TripletMarginLoss] Triplet margin loss (L2 anchor-positive and anchor-negative distances, hinge, mean), with the embedding dim at 4000 instead of 4096 so it is not a multiple of 128."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': '99_TripletMarginLoss_m1_ragged',
    'batch_size': 4096,
    'dim': 4000,
    'margin': 1.0,
    'p': 2,
    'eps': 1e-6,
}


def create_inputs(dtype=jnp.float32):
    k_scale, k_a, k_p, k_n = jax.random.split(jax.random.key(0), 4)
    shape = (CONFIG['batch_size'], CONFIG['dim'])
    scale = jax.random.uniform(k_scale, (), dtype=dtype)
    anchor = jax.random.uniform(k_a, shape, dtype=dtype) * scale
    positive = jax.random.uniform(k_p, shape, dtype=dtype)
    negative = jax.random.uniform(k_n, shape, dtype=dtype)
    return anchor, positive, negative


def _pairwise_distance(x1, x2):
    diff = x1 - x2 + CONFIG['eps']
    return jnp.sqrt(jnp.sum(diff * diff, axis=-1))


def workload(anchor, positive, negative):
    d_ap = _pairwise_distance(anchor, positive)
    d_an = _pairwise_distance(anchor, negative)
    losses = jnp.maximum(d_ap - d_an + CONFIG['margin'], 0.0)
    return jnp.mean(losses)
