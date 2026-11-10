"""[bf16 variant] QuadricLinearLoss: subtract -> abs -> clamp(max=delta) -> quadratic + linear combine -> weighted mean."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'QuadricLinearLoss_14407',
    'batch': 8,
    'height': 128,
    'width': 128,
    'depth': 128,
    'clip_delta': 4.0,
    'quadratic_coeff': 0.5,
}


def create_inputs(dtype=jnp.bfloat16):
    shape = (CONFIG['batch'], CONFIG['height'], CONFIG['width'], CONFIG['depth'])
    key = jax.random.key(0)
    k_pred, k_true, k_weights = jax.random.split(key, 3)
    y_pred = jax.random.normal(k_pred, shape, dtype)
    y_true = jax.random.normal(k_true, shape, dtype)
    weights = jax.random.uniform(k_weights, shape, dtype)
    return y_pred, y_true, weights


def workload(y_pred, y_true, weights):
    clip_delta = CONFIG['clip_delta']
    td_error = y_true - y_pred
    td_error_abs = jnp.abs(td_error)
    quadratic_part = jnp.minimum(td_error_abs, clip_delta)
    linear_part = td_error_abs - quadratic_part
    loss = CONFIG['quadratic_coeff'] * quadratic_part ** 2 + clip_delta * linear_part
    return jnp.mean(loss * weights)
