"""Kullback-Leibler divergence loss (batchmean reduction) on probability distributions."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': '98_KLDivLoss',
    'batch_size': 8192,
    'features': 8192,
}

def create_inputs(dtype=jnp.float32):
    k1, k2, k3 = jax.random.split(jax.random.key(0), 3)
    shape = (CONFIG['batch_size'], CONFIG['features'])
    scale = jax.random.uniform(k1, (), dtype=dtype)
    preds_logits = jax.random.uniform(k2, shape, dtype=dtype) * scale
    targets_logits = jax.random.uniform(k3, shape, dtype=dtype)
    predictions = jax.nn.softmax(preds_logits, axis=-1)
    targets = jax.nn.softmax(targets_logits, axis=-1)
    return (predictions, targets)

def workload(predictions, targets):
    log_predictions = jnp.log(predictions)
    pointwise = jnp.where(targets > 0, targets * (jnp.log(targets) - log_predictions), 0.0)
    return jnp.sum(pointwise) / predictions.shape[0]
