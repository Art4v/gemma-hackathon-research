"""[bf16 variant] Differencial SMAPE: sum(|p - t| / clamp(|t| + |p| + eps, min=0.5 + eps) * 2)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'Differencial_SMAPE_7462',
    'shape': (8, 64, 256, 256),
    'epsilon': 0.1,
    'clamp_base': 0.5,
    'scale': 2.0,
}


def create_inputs(dtype=jnp.bfloat16):
    k_true, k_pred = jax.random.split(jax.random.key(0))
    true = jax.random.uniform(k_true, CONFIG['shape'], dtype)
    predicted = jax.random.uniform(k_pred, CONFIG['shape'], dtype)
    return true, predicted


def workload(true, predicted):
    eps = CONFIG['epsilon']
    summ = jnp.maximum(jnp.abs(true) + jnp.abs(predicted) + eps, CONFIG['clamp_base'] + eps)
    smape = jnp.abs(predicted - true) / summ * CONFIG['scale']
    return jnp.sum(smape)
