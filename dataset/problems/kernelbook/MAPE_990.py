"""[bf16 variant] MAPE: mean(100 * |pred - actual| / max(pred, actual))."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'MAPE_990',
    'shape': (16, 64, 128, 128),
    'scale': 100.0,
}


def create_inputs(dtype=jnp.bfloat16):
    k_pred, k_actual = jax.random.split(jax.random.key(0))
    shape = CONFIG['shape']
    pred = jax.random.uniform(k_pred, shape, dtype)
    actual = jax.random.uniform(k_actual, shape, dtype)
    return pred, actual


def workload(pred, actual):
    mape = CONFIG['scale'] * jnp.abs(pred - actual) / jnp.maximum(pred, actual)
    return jnp.mean(mape)
