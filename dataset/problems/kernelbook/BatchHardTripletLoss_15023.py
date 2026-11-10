"""Batch-hard triplet loss: pairwise L2 distance -> label mask -> hardest pos/neg -> margin ranking loss."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'BatchHardTripletLoss_15023',
    'batch': 8192,
    'dim': 4096,
    'num_classes': 128,
    'margin': 0.0,
    'clamp_min': 1e-12,
}


def create_inputs(dtype=jnp.float32):
    k_x, k_perm = jax.random.split(jax.random.key(0))
    b, d = CONFIG['batch'], CONFIG['dim']
    inputs = jax.random.uniform(k_x, (b, d), dtype)
    # Balanced classes keep torch's dist[mask].view(B, -1) well defined.
    labels = jnp.arange(b, dtype=jnp.int32) % CONFIG['num_classes']
    targets = jax.random.permutation(k_perm, labels)
    return inputs, targets


def workload(inputs, targets):
    b = inputs.shape[0]
    sq = jnp.sum(inputs * inputs, axis=1, keepdims=True)
    gram = jnp.matmul(inputs, inputs.T, precision=lax.Precision.HIGHEST)
    dist = jnp.sqrt(jnp.maximum(sq + sq.T - 2.0 * gram, CONFIG['clamp_min']))
    t = jnp.broadcast_to(targets, (b, b))
    mask = t == t.T
    dist_ap = jnp.max(jnp.where(mask, dist, -jnp.inf), axis=1)
    dist_an = jnp.min(jnp.where(mask, jnp.inf, dist), axis=1)
    # MarginRankingLoss(x1=dist_an, x2=dist_ap, y=1), mean reduction.
    return jnp.mean(jnp.maximum(0.0, dist_ap - dist_an + CONFIG['margin']))
