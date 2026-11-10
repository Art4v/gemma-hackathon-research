"""[bf16 variant] Contrastive embedding loss: diff -> sum of squares over dim 1 -> sqrt -> margin clamp -> weighted squared terms -> mean reduction."""
import jax
import jax.numpy as jnp

# B must equal C: PyTorch broadcasts distance_pred (B,H,W) against distance_true (B,C,H,W) right-aligned.
CONFIG = {
    'name': 'ContrastiveEmbeddingLoss_3183',
    'batch': 128,
    'channels': 128,
    'height': 8,
    'width': 128,
    'margin': 1.0,
    'reduction': 'mean',
}


def create_inputs(dtype=jnp.bfloat16):
    shape = (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width'])
    k1, k2, k3 = jax.random.split(jax.random.key(0), 3)
    embeddings_left = jax.random.uniform(k1, shape, dtype)
    embeddings_right = jax.random.uniform(k2, shape, dtype)
    distance_true = jax.random.uniform(k3, shape, dtype)
    return embeddings_left, embeddings_right, distance_true


def workload(embeddings_left, embeddings_right, distance_true):
    diff = embeddings_left - embeddings_right
    distance_pred = jnp.sqrt(jnp.sum(diff ** 2, axis=1))
    bs = distance_true.shape[0]
    margin_distance = jnp.maximum(CONFIG['margin'] - distance_pred, 0.0)
    loss = (1 - distance_true) * distance_pred ** 2 + distance_true * margin_distance ** 2
    if CONFIG['reduction'] == 'mean':
        return jnp.sum(loss) / 2.0 / bs
    if CONFIG['reduction'] == 'sum':
        return jnp.sum(loss)
    return loss
