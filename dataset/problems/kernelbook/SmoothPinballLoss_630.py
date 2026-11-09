"""[bf16 variant] SmoothPinballLoss: q*(target-pred) + softplus(pred-target, beta=1/alpha), sum over dim 1, then mean."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'SmoothPinballLoss_630',
    'batch': 16,
    'channels': 128,
    'height': 128,
    'width': 128,
    'quantiles': 4.0,
    'alpha': 0.001,
    'softplus_threshold': 20.0,
}


def create_inputs(dtype=jnp.bfloat16):
    shape = (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width'])
    k_pred, k_target = jax.random.split(jax.random.key(0))
    pred = jax.random.uniform(k_pred, shape, dtype)
    target = jax.random.uniform(k_target, shape, dtype)
    return pred, target


def workload(pred, target):
    error = target - pred
    q_error = CONFIG['quantiles'] * error
    beta = 1.0 / CONFIG['alpha']
    z = -error * beta
    # Match torch softplus(x, beta, threshold): linear above threshold.
    soft_error = jnp.where(z > CONFIG['softplus_threshold'], -error, jax.nn.softplus(z) / beta)
    losses = q_error + soft_error
    return jnp.mean(jnp.sum(losses, axis=1))
