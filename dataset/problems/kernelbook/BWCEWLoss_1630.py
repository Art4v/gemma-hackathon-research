"""BCEWithLogitsLoss (mean reduction) -> mean: binary weighted cross entropy loss with default settings."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'BWCEWLoss_1630',
    'batch': 8,
    'channels': 32,
    'height': 512,
    'width': 512,
    'robust_lambda': 0,
    'confidence_penalty': 0,
}


def create_inputs(dtype=jnp.float32):
    k1, k2 = jax.random.split(jax.random.key(0))
    shape = (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width'])
    preds = jax.random.uniform(k1, shape, dtype)
    target = jax.random.uniform(k2, shape, dtype)
    return preds, target


def workload(preds, target):
    # ponytail: robust_lambda and confidence_penalty are 0 by default, so their branches are dead and omitted.
    x = preds.astype(jnp.float32)
    t = target.astype(jnp.float32)
    loss = jnp.maximum(x, 0.0) - x * t + jnp.log1p(jnp.exp(-jnp.abs(x)))
    return jnp.mean(loss)
