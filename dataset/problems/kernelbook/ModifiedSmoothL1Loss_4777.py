"""ModifiedSmoothL1Loss: abs difference -> piecewise quadratic/linear smooth L1 -> zero below dead-zone threshold."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'ModifiedSmoothL1Loss_4777',
    'shape': (8, 128, 128, 256),
    'alpha': 4,
    'zero_threshold': 0.01,
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    k_targets, k_pred = jax.random.split(key)
    normed_targets = jax.random.uniform(k_targets, CONFIG['shape'], dtype)
    pos_reg = jax.random.uniform(k_pred, CONFIG['shape'], dtype)
    return normed_targets, pos_reg


def workload(normed_targets, pos_reg):
    alpha = CONFIG['alpha']
    regression_diff = jnp.abs(normed_targets - pos_reg)
    regression_loss = jnp.where(
        regression_diff <= 1.0 / alpha,
        0.5 * alpha * jnp.power(regression_diff, 2),
        regression_diff - 0.5 / alpha,
    )
    regression_loss = jnp.where(
        regression_diff <= CONFIG['zero_threshold'],
        jnp.zeros_like(regression_loss),
        regression_loss,
    )
    return regression_loss
