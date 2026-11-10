"""[bf16 variant] BCEFocalLoss: BCE-with-logits -> sigmoid -> asymmetric focal weighting -> mean."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'BCEFocalLoss_7049',
    'batch': 16,
    'channels': 128,
    'height': 128,
    'width': 128,
    'alpha': 0.25,
    'gamma': 2.0,
}


def create_inputs(dtype=jnp.bfloat16):
    shape = (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width'])
    key = jax.random.key(0)
    k_preds, k_targets = jax.random.split(key)
    preds = jax.random.uniform(k_preds, shape, dtype)
    targets = jax.random.uniform(k_targets, shape, dtype)
    return preds, targets


def workload(preds, targets):
    alpha = CONFIG['alpha']
    gamma = CONFIG['gamma']
    bce_loss = (1.0 - targets) * preds - jax.nn.log_sigmoid(preds)
    probas = jax.nn.sigmoid(preds)
    loss = (
        targets * alpha * jnp.power(1.0 - probas, gamma) * bce_loss
        + (1.0 - targets) * jnp.power(probas, gamma) * bce_loss
    )
    return jnp.mean(loss)
