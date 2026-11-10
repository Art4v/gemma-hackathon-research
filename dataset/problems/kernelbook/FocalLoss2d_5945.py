"""Clamp -> binary masks -> detached focal weights -> weighted log terms -> mean (FocalLoss2d)."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'FocalLoss2d_5945',
    'batch': 8,
    'channels': 128,
    'height': 128,
    'width': 128,
    'alpha': 0.25,
    'gamma': 2,
    'smooth': 1e-06,
}


def create_inputs(dtype=jnp.float32):
    shape = (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width'])
    key = jax.random.key(0)
    k_prob, k_target = jax.random.split(key)
    prob = jax.random.uniform(k_prob, shape, dtype)
    target = jax.random.randint(k_target, shape, 0, 2).astype(dtype)
    return prob, target


def workload(prob, target):
    alpha = CONFIG['alpha']
    gamma = CONFIG['gamma']
    smooth = CONFIG['smooth']
    prob = jnp.clip(prob, smooth, 1.0 - smooth)
    pos_mask = (target == 1).astype(prob.dtype)
    neg_mask = (target == 0).astype(prob.dtype)
    pos_weight = lax.stop_gradient(pos_mask * jnp.power(1 - prob, gamma))
    pos_loss = -alpha * (pos_weight * jnp.log(prob))
    neg_weight = lax.stop_gradient(neg_mask * jnp.power(prob, gamma))
    neg_loss = -(1 - alpha) * (neg_weight * jnp.log(1 - prob))
    loss = pos_loss + neg_loss
    return jnp.mean(loss)
