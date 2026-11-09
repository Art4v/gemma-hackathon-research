"""Sigmoid -> flatten -> BCELoss (mean reduction, log clamped at -100)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'BCELoss2d_1686',
    'batch': 8,
    'channels': 64,
    'height': 256,
    'width': 256,
    'log_clamp': -100.0,
}


def create_inputs(dtype=jnp.float32):
    shape = (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width'])
    k_logits, k_targets = jax.random.split(jax.random.key(0))
    logits = jax.random.uniform(k_logits, shape, dtype)
    targets = jax.random.uniform(k_targets, shape, dtype)
    return logits, targets


def workload(logits, targets):
    probs = jax.nn.sigmoid(logits).reshape(-1)
    t = targets.reshape(-1)
    # PyTorch BCELoss clamps each log term to >= -100 to avoid -inf.
    log_p = jnp.maximum(jnp.log(probs), CONFIG['log_clamp'])
    log_1mp = jnp.maximum(jnp.log(1.0 - probs), CONFIG['log_clamp'])
    return -jnp.mean(t * log_p + (1.0 - t) * log_1mp)
