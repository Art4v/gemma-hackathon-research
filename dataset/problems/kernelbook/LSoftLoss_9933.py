"""LSoftLoss: stop_gradient(beta * y_true + (1 - beta) * y_pred) soft target followed by element-wise binary cross entropy with log clamp."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'LSoftLoss_9933',
    'batch': 16,
    'channels': 128,
    'height': 128,
    'width': 128,
    'log_min': -100.0,
}


def create_inputs(dtype=jnp.float32):
    shape = (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width'])
    k_pred, k_true, k_beta = jax.random.split(jax.random.key(0), 3)
    y_pred = jax.random.uniform(k_pred, shape, dtype)
    y_true = jax.random.uniform(k_true, shape, dtype)
    beta = jax.random.uniform(k_beta, shape, dtype)
    return y_pred, y_true, beta


def workload(y_pred, y_true, beta):
    target = lax.stop_gradient(beta * y_true + (1 - beta) * y_pred)
    log_x = jnp.maximum(jnp.log(y_pred), CONFIG['log_min'])
    log_1mx = jnp.maximum(jnp.log(1 - y_pred), CONFIG['log_min'])
    return (target - 1) * log_1mx - target * log_x
