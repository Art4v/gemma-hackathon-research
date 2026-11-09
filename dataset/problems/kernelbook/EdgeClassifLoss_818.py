"""Sigmoid -> BCELoss(mean) edge classification loss."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'EdgeClassifLoss_818',
    'batch': 16,
    'channels': 4,
    'n_rows': 1024,
    'n_cols': 1024,
    'log_clamp': -100.0,
}


def create_inputs(dtype=jnp.float32):
    shape = (CONFIG['batch'], CONFIG['channels'], CONFIG['n_rows'], CONFIG['n_cols'])
    k1, k2 = jax.random.split(jax.random.key(0))
    raw_scores = jax.random.uniform(k1, shape, dtype)
    target = jax.random.uniform(k2, shape, dtype)
    return raw_scores, target


def workload(raw_scores, target):
    preds = jax.nn.sigmoid(raw_scores)
    # PyTorch BCELoss clamps log outputs at -100 to avoid -inf.
    log_p = jnp.maximum(jnp.log(preds), CONFIG['log_clamp'])
    log_1mp = jnp.maximum(jnp.log(1.0 - preds), CONFIG['log_clamp'])
    loss = -(target * log_p + (1.0 - target) * log_1mp)
    return jnp.mean(loss)
