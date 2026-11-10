"""BCEBlurWithLogitsLoss: BCE-with-logits -> sigmoid -> exp blur factor -> elementwise product -> mean."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'BCEBlurWithLogitsLoss_4807',
    'batch': 16,
    'channels': 128,
    'height': 128,
    'width': 128,
    'alpha': 0.05,
    'eps': 0.0001,
}


def create_inputs(dtype=jnp.float32):
    shape = (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width'])
    key = jax.random.key(0)
    k_pred, k_true = jax.random.split(key)
    pred = jax.random.uniform(k_pred, shape, dtype)
    true = jax.random.uniform(k_true, shape, dtype)
    return pred, true


def workload(pred, true):
    loss = (1 - true) * pred + jax.nn.softplus(-pred)
    dx = jax.nn.sigmoid(pred) - true
    alpha_factor = 1 - jnp.exp((dx - 1) / (CONFIG['alpha'] + CONFIG['eps']))
    return jnp.mean(loss * alpha_factor)
