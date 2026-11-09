"""[bf16 variant] MSE loss: sum((real - pred)^2) / numel."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'MSE_285',
    'batch': 16,
    'channels': 32,
    'height': 256,
    'width': 256,
}


def create_inputs(dtype=jnp.bfloat16):
    shape = (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width'])
    k_pred, k_real = jax.random.split(jax.random.key(0))
    pred = jax.random.uniform(k_pred, shape, dtype)
    real = jax.random.uniform(k_real, shape, dtype)
    return pred, real


def workload(pred, real):
    diffs = real - pred
    return jnp.sum(diffs ** 2) / diffs.size
