"""[bf16 variant] L2 normalize along channel axis -> pairwise dot product -> negated mean (cosine angular loss)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'CosineAngularLoss_16281',
    'batch': 8,
    'channels': 128,
    'height': 128,
    'width': 128,
    'axis': 1,
    'eps': 1e-12,
}


def create_inputs(dtype=jnp.bfloat16):
    shape = (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width'])
    k_preds, k_truths = jax.random.split(jax.random.key(0))
    preds = jax.random.uniform(k_preds, shape, dtype)
    truths = jax.random.uniform(k_truths, shape, dtype)
    return preds, truths


def _normalize(v):
    axis = CONFIG['axis']
    norm = jnp.sqrt(jnp.sum(v * v, axis=axis, keepdims=True))
    return v / jnp.maximum(norm, CONFIG['eps'])


def workload(preds, truths):
    preds_norm = _normalize(preds)
    truths_norm = _normalize(truths)
    return jnp.mean(-jnp.sum(preds_norm * truths_norm, axis=CONFIG['axis']))
