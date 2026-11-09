"""Weighted MAE with threshold-binned weights: where(target bins) * |predict - target| -> mean."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'weighted_mae_windows_9647',
    'batch': 16,
    'length': 16,
    'height': 256,
    'width': 256,
    'weights': (0.5, 1.2, 1.4, 1.6, 1.8, 2.0),
    'thresholds': (5.0, 15.0, 30.0, 40.0, 45.0),
    'value_min': 0.0,
    'value_max': 60.0,
}


def create_inputs(dtype=jnp.float32):
    shape = (CONFIG['batch'], CONFIG['length'], CONFIG['height'], CONFIG['width'])
    k_pred, k_tgt = jax.random.split(jax.random.key(0))
    # ponytail: range spans every threshold bin so all weights are exercised.
    predict = jax.random.uniform(k_pred, shape, dtype,
                                 CONFIG['value_min'], CONFIG['value_max'])
    target = jax.random.uniform(k_tgt, shape, dtype,
                                CONFIG['value_min'], CONFIG['value_max'])
    return predict, target


def workload(predict, target):
    weights = CONFIG['weights']
    # Thresholds are ascending, so successive overwrites reproduce the torch masked fills.
    balance = jnp.full_like(target, weights[0])
    for th, w in zip(CONFIG['thresholds'], weights[1:]):
        balance = jnp.where(target >= th, jnp.asarray(w, target.dtype), balance)
    return jnp.mean(balance * jnp.abs(predict - target))
