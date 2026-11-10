"""FocalLoss: clamp -> binary cross-entropy -> alpha-balanced focal weighting -> sum."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'FocalLoss_9305',
    'shape': (16, 32, 256, 256),
    'gamma': 2,
    'alpha': 0.25,
    'eps': 1e-07,
}


def create_inputs(dtype=jnp.float32):
    k1, k2 = jax.random.split(jax.random.key(0))
    x = jax.random.uniform(k1, CONFIG['shape'], dtype)
    target = jax.random.uniform(k2, CONFIG['shape'], dtype)
    return x, target


def workload(x, target):
    eps = CONFIG['eps']
    alpha = CONFIG['alpha']
    gamma = CONFIG['gamma']
    x = jnp.clip(x, eps, 1.0 - eps)
    logpt = target * jnp.log(x) + (1 - target) * jnp.log(1 - x)
    pt = jnp.exp(logpt)
    at = alpha * target + (1 - alpha) * (1 - target)
    focal = -at * logpt * (1 - pt) ** gamma
    return jnp.sum(focal)
