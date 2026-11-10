"""PartialBCELoss: log_sigmoid BCE terms -> mask multiply -> sum(dim=-1) -> mean."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'PartialBCELoss_8169',
    'batch': 16,
    'channels': 32,
    'height': 256,
    'width': 256,
}


def create_inputs(dtype=jnp.float32):
    shape = (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width'])
    k1, k2, k3 = jax.random.split(jax.random.key(0), 3)
    logits = jax.random.uniform(k1, shape, dtype)
    targets = jax.random.uniform(k2, shape, dtype)
    targets_mask = jax.random.uniform(k3, shape, dtype)
    return logits, targets, targets_mask


def workload(logits, targets, targets_mask):
    # ponytail: weights=None path only, matching get_inputs().
    pos_vals = -targets * jax.nn.log_sigmoid(logits)
    neg_vals = -jax.nn.log_sigmoid(-logits) * (1 - targets)
    losses = jnp.sum((pos_vals + neg_vals) * targets_mask, axis=-1)
    return jnp.mean(losses)
