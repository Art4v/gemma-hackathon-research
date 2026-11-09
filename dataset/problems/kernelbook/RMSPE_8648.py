"""RMSPE: sqrt(mean(square(|pred - target| / (|target| + eps))))."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'RMSPE_8648',
    'batch': 16,
    'channels': 32,
    'height': 256,
    'width': 256,
    'eps': 1e-8,
}


def create_inputs(dtype=jnp.float32):
    k_pred, k_target = jax.random.split(jax.random.key(0), 2)
    shape = (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width'])
    pred = jax.random.uniform(k_pred, shape, dtype)
    target = jax.random.uniform(k_target, shape, dtype)
    return pred, target


def workload(pred, target):
    ratio = jnp.abs(pred - target) / (jnp.abs(target) + CONFIG['eps'])
    return jnp.sqrt(jnp.mean(jnp.square(ratio)))
