"""[bf16 variant] Sigmoid focal loss: BCEWithLogits -> sigmoid p_t -> (1-p_t)^gamma * alpha weight -> mean."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'FocalLoss_9710',
    'batch': 16,
    'num_anchors': 4096,
    'num_classes': 512,
    'gamma': 2.0,
    'alpha': 0.25,
}


def create_inputs(dtype=jnp.bfloat16):
    k_pred, k_tgt = jax.random.split(jax.random.key(0))
    shape = (CONFIG['batch'], CONFIG['num_anchors'], CONFIG['num_classes'])
    prediction = jax.random.uniform(k_pred, shape, dtype)
    target = jax.random.uniform(k_tgt, shape, dtype)
    return prediction, target


def workload(prediction, target):
    gamma = CONFIG['gamma']
    alpha = CONFIG['alpha']
    x, t = prediction, target
    # Same stable formulation as torch BCEWithLogitsLoss.
    bce = jnp.maximum(x, 0.0) - x * t + jnp.log1p(jnp.exp(-jnp.abs(x)))
    p = jax.nn.sigmoid(x)
    p_t = t * p + (1.0 - t) * (1.0 - p)
    modulating = jnp.power(1.0 - p_t, gamma)
    alpha_weight = t * alpha + (1.0 - t) * (1.0 - alpha)
    return jnp.mean(modulating * alpha_weight * bce)
