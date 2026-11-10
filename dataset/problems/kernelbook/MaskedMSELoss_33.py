"""Masked MSE loss: sum((pred * mask - target)^2) / sum(mask)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'MaskedMSELoss_33',
    'shape': (16, 128, 128, 128),
}


def create_inputs(dtype=jnp.float32):
    k_pred, k_target, k_mask = jax.random.split(jax.random.key(0), 3)
    shape = CONFIG['shape']
    pred = jax.random.uniform(k_pred, shape, dtype)
    target = jax.random.uniform(k_target, shape, dtype)
    mask = jax.random.uniform(k_mask, shape, dtype)
    return pred, target, mask


def workload(pred, target, mask):
    diff = pred * mask - target
    return jnp.sum(diff * diff) / jnp.sum(mask)
