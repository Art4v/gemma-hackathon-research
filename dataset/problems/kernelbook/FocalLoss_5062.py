"""Focal loss: flatten -> binary cross entropy (mean) -> alpha * (1 - exp(-BCE))**gamma * BCE."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'FocalLoss_5062',
    'batch': 8,
    'channels': 64,
    'height': 256,
    'width': 256,
    'alpha': 0.8,
    'gamma': 2,
    'log_clamp': -100.0,
}


def create_inputs(dtype=jnp.float32):
    shape = (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width'])
    k_in, k_tgt = jax.random.split(jax.random.key(0))
    inputs = jax.random.uniform(k_in, shape, dtype)
    targets = jax.random.uniform(k_tgt, shape, dtype)
    return inputs, targets


def workload(inputs, targets):
    x = inputs.reshape(-1)
    t = targets.reshape(-1)
    # PyTorch BCE clamps each log term to >= -100 so x=0 or x=1 stays finite.
    log_x = jnp.maximum(jnp.log(x), CONFIG['log_clamp'])
    log_1mx = jnp.maximum(jnp.log(1.0 - x), CONFIG['log_clamp'])
    bce = jnp.mean(-(t * log_x + (1.0 - t) * log_1mx))
    bce_exp = jnp.exp(-bce)
    return CONFIG['alpha'] * (1.0 - bce_exp) ** CONFIG['gamma'] * bce
