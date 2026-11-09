"""ATLoss: masked log_softmax (last axis) x labels, sum over axis 1, mean (adaptive-threshold loss)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'ATLoss_4294',
    'batch': 8,
    'channels': 128,
    'height': 128,
    'width': 256,
    'mask_value': 1e30,
}


def create_inputs(dtype=jnp.float32):
    shape = (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width'])
    k_logits, k_labels = jax.random.split(jax.random.key(0))
    logits = jax.random.uniform(k_logits, shape, dtype)
    labels = jax.random.uniform(k_labels, shape, dtype)
    return logits, labels


def workload(logits, labels):
    big = CONFIG['mask_value']
    th_label = jnp.zeros_like(labels).at[:, 0].set(1.0)
    labels = labels.at[:, 0].set(0.0)
    p_mask = labels + th_label
    n_mask = 1.0 - labels
    logit1 = logits - (1.0 - p_mask) * big
    loss1 = -(jax.nn.log_softmax(logit1, axis=-1) * labels).sum(axis=1)
    logit2 = logits - (1.0 - n_mask) * big
    loss2 = -(jax.nn.log_softmax(logit2, axis=-1) * th_label).sum(axis=1)
    return jnp.mean(loss1 + loss2)
