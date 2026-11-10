"""Weighted BCE loss: log_sigmoid -> weighted BCE -> sum(last dim)/size(1) -> mean."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'WeightedBCELoss_8181',
    'shape': (8, 128, 128, 128),
    'neg_scale': -1,
    'bce_sum': False,
}


def create_inputs(dtype=jnp.float32):
    k1, k2, k3 = jax.random.split(jax.random.key(0), 3)
    shape = CONFIG['shape']
    logits = jax.random.uniform(k1, shape, dtype)
    targets = jax.random.uniform(k2, shape, dtype)
    target_weights = jax.random.uniform(k3, shape, dtype)
    return logits, targets, target_weights


def workload(logits, targets, target_weights):
    neg_vals = jax.nn.log_sigmoid(-logits) * (1 - targets)
    if CONFIG['neg_scale'] > 0:
        neg_vals = neg_vals * CONFIG['neg_scale']
    vals = -targets * jax.nn.log_sigmoid(logits) - neg_vals
    losses = jnp.sum(vals * target_weights, axis=-1)
    if not CONFIG['bce_sum']:
        losses = losses / logits.shape[1]
    return jnp.mean(losses)
