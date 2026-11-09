"""LossCrossentropyAgg: clamp(preds) * target -> mean over dim 1 -> -log."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'LossCrossentropyAgg_3763',
    'batch': 8,
    'channels': 128,
    'height': 128,
    'width': 256,
    'clamp_min': 1e-10,
    'clamp_max': 1.0,
    'reduce_dim': 1,
}


def create_inputs(dtype=jnp.float32):
    shape = (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width'])
    k_preds, k_target = jax.random.split(jax.random.key(0))
    preds = jax.random.uniform(k_preds, shape, dtype)
    target = jax.random.uniform(k_target, shape, dtype)
    return preds, target


def workload(preds, target):
    preds = jnp.clip(preds, CONFIG['clamp_min'], CONFIG['clamp_max'])
    return -jnp.log(jnp.mean(target * preds, axis=CONFIG['reduce_dim']))
