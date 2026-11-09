"""MSE: sum((real - pred)^2) / numel."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'MSE_3948',
    'batch': 16,
    'd1': 128,
    'd2': 128,
    'd3': 256,
}


def create_inputs(dtype=jnp.float32):
    shape = (CONFIG['batch'], CONFIG['d1'], CONFIG['d2'], CONFIG['d3'])
    k_pred, k_real = jax.random.split(jax.random.key(0))
    pred = jax.random.uniform(k_pred, shape, dtype)
    real = jax.random.uniform(k_real, shape, dtype)
    return pred, real


def workload(pred, real):
    diffs = real - pred
    return jnp.sum(diffs ** 2) / diffs.size
