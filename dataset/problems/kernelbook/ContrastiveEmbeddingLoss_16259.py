"""Contrastive embedding loss: diff -> L2 distance over axis 1 -> margin clamp -> broadcast weighted squared sum -> mean over batch."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'ContrastiveEmbeddingLoss_16259',
    'batch': 128,
    'channels': 128,  # must equal batch: distance_pred (B,H,W) broadcasts against (B,C,H,W) trailing axes, as in PyTorch
    'height': 16,
    'width': 128,
    'margin': 1.0,
}


def create_inputs(dtype=jnp.float32):
    shape = (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width'])
    k1, k2, k3 = jax.random.split(jax.random.key(0), 3)
    embeddings_left = jax.random.uniform(k1, shape, dtype)
    embeddings_right = jax.random.uniform(k2, shape, dtype)
    distance_true = jax.random.uniform(k3, shape, dtype)
    return embeddings_left, embeddings_right, distance_true


def workload(embeddings_left, embeddings_right, distance_true):
    diff = embeddings_left - embeddings_right
    distance_pred = jnp.sqrt(jnp.sum(diff * diff, axis=1))
    margin_distance = jnp.maximum(CONFIG['margin'] - distance_pred, 0.0)
    loss = (1.0 - distance_true) * distance_pred ** 2 + distance_true * margin_distance ** 2
    bs = distance_true.shape[0]
    return jnp.sum(loss) / 2.0 / bs
