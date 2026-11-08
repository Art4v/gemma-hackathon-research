"""Smooth L1 (Huber) loss: elementwise |p - t| -> piecewise quadratic/linear -> mean."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': '96_HuberLoss',
    'batch_size': 8192,
    'dim': 8192,
    'beta': 1.0,
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    k_scale, k_pred, k_targ = jax.random.split(key, 3)
    shape = (CONFIG['batch_size'], CONFIG['dim'])
    scale = jax.random.uniform(k_scale, (), dtype=jnp.float32)
    predictions = (jax.random.uniform(k_pred, shape, dtype=jnp.float32) * scale).astype(dtype)
    targets = jax.random.uniform(k_targ, shape, dtype=dtype)
    return predictions, targets


def workload(predictions, targets):
    beta = CONFIG['beta']
    diff = jnp.abs(predictions.astype(jnp.float32) - targets.astype(jnp.float32))
    loss = jnp.where(diff < beta, 0.5 * diff * diff / beta, diff - 0.5 * beta)
    return jnp.mean(loss).astype(predictions.dtype)
