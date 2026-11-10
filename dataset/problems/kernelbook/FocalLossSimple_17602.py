"""Sigmoid focal loss: sigmoid -> BCE-with-logits -> (1-p_t)^gamma * alpha_t * ce -> mean."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'FocalLossSimple_17602',
    'batch': 8,
    'channels': 64,
    'height': 256,
    'width': 256,
    'gamma': 2,
    'alpha': 0.25,
}


def create_inputs(dtype=jnp.float32):
    k1, k2 = jax.random.split(jax.random.key(0))
    shape = (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width'])
    logit = jax.random.uniform(k1, shape, dtype)
    target = jax.random.uniform(k2, shape, dtype)
    return logit, target


def workload(logit, target):
    gamma = CONFIG['gamma']
    alpha = CONFIG['alpha']
    target = target.astype(logit.dtype)
    pred_prob = jax.nn.sigmoid(logit)
    ce = jnp.maximum(logit, 0) - logit * target + jnp.log1p(jnp.exp(-jnp.abs(logit)))
    p_t = target * pred_prob + (1 - target) * (1 - pred_prob)
    modulating_factor = jnp.power(1.0 - p_t, gamma)
    alpha_factor = target * alpha + (1 - target) * (1 - alpha)
    loss = alpha_factor * modulating_factor * ce
    return jnp.mean(loss)
