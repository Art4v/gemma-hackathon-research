"""Focal loss V1: sigmoid -> pt select -> BCE with logits -> alpha * (1 - pt)^gamma weighting -> mean reduction."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'FocalLossV1_6875',
    'batch': 8,
    'channels': 128,
    'height': 128,
    'width': 128,
    'alpha': 0.25,
    'gamma': 2,
    'reduction': 'mean',
}


def create_inputs(dtype=jnp.float32):
    shape = (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width'])
    key = jax.random.key(0)
    k_logits, k_label = jax.random.split(key)
    logits = jax.random.uniform(k_logits, shape, dtype)
    label = jax.random.uniform(k_label, shape, dtype)
    return logits, label


def workload(logits, label):
    logits = logits.astype(jnp.float32)
    is_positive = label == 1
    alpha = jnp.where(
        is_positive,
        jnp.full_like(logits, CONFIG['alpha']),
        jnp.full_like(logits, 1 - CONFIG['alpha']),
    )
    probs = jax.nn.sigmoid(logits)
    pt = jnp.where(is_positive, probs, 1 - probs)
    target = label.astype(jnp.float32)
    ce_loss = (1 - target) * logits - jax.nn.log_sigmoid(logits)
    loss = alpha * jnp.power(1 - pt, CONFIG['gamma']) * ce_loss
    if CONFIG['reduction'] == 'mean':
        loss = jnp.mean(loss)
    if CONFIG['reduction'] == 'sum':
        loss = jnp.sum(loss)
    return loss
