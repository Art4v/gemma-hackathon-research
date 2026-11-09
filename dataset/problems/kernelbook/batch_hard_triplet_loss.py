"""[bf16 variant] Batch-all triplet loss: pairwise euclidean distance -> masked triplet hinge -> mean over positive triplets."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'batch_hard_triplet_loss',
    'batch': 1024,
    'embed_dim': 16384,
    'num_classes': 64,
    'margin': 0.1,
    'dist_eps': 1e-16,
    'hard_threshold': 1e-16,
    'denom_eps': 1e-16,
}


def create_inputs(dtype=jnp.bfloat16):
    k_emb, k_lab = jax.random.split(jax.random.key(0))
    embeddings = jax.random.uniform(k_emb, (CONFIG['batch'], CONFIG['embed_dim']), dtype)
    labels = jax.random.randint(k_lab, (CONFIG['batch'],), 0, CONFIG['num_classes'], dtype=jnp.int32)
    return embeddings, labels


def _pairwise_distance(x):
    cor = jnp.matmul(x, x.T, precision=lax.Precision.HIGHEST)
    norm = jnp.diagonal(cor)
    d = jax.nn.relu(norm[:, None] - 2.0 * cor + norm[None, :])
    zero = (d == 0.0).astype(d.dtype)
    return jnp.sqrt(d + zero * CONFIG['dist_eps']) * (1.0 - zero)


def _triplet_mask(labels):
    # Index the same way torch unsqueezes, so any label rank broadcasts exactly as in torch.
    not_eye = ~jnp.eye(labels.shape[0], dtype=bool)
    distinct = not_eye[:, :, None] & not_eye[:, None, :] & not_eye[None, :, :]
    label_equal = labels[None] == labels[:, None]
    valid = ~label_equal[:, None] & label_equal[:, :, None]
    return valid & distinct


def workload(embeddings, labels):
    d = _pairwise_distance(embeddings)
    loss = d[:, :, None] - d[:, None, :] + CONFIG['margin']
    mask = _triplet_mask(labels).astype(loss.dtype)
    triplet_loss = jax.nn.relu(loss * mask)
    num_hard = jnp.sum((triplet_loss > CONFIG['hard_threshold']).astype(loss.dtype))
    return jnp.sum(triplet_loss) / (num_hard + CONFIG['denom_eps'])
