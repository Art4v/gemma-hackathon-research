"""Binary focal loss: log-sigmoid terms, sigmoid power weights, weight-normalized mean."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'FocalLoss_5147',
    'batch': 16,
    'channels': 128,
    'height': 128,
    'width': 128,
    'gamma': 2,
    'zero_clamp_scale': 0.5,
}


def create_inputs(dtype=jnp.float32):
    shape = (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width'])
    key_input, key_target = jax.random.split(jax.random.key(0))
    x = jax.random.uniform(key_input, shape, dtype)
    target = jax.random.uniform(key_target, shape, dtype)
    return x, target


def workload(x, target):
    gamma = CONFIG['gamma']
    log_term = jnp.log(1 + jnp.exp(-jnp.abs(x)))
    zero_term = CONFIG['zero_clamp_scale'] * jnp.clip(x, 0, 0)
    pos_log_sig = jnp.minimum(x, 0) - log_term + zero_term
    neg_log_sig = jnp.minimum(-x, 0) - log_term + zero_term
    prob = jax.nn.sigmoid(x)
    pos_weight = jnp.power(1 - prob, gamma)
    neg_weight = jnp.power(prob, gamma)
    loss = -(target * pos_weight * pos_log_sig
             + (1 - target) * neg_weight * neg_log_sig)
    avg_weight = target * pos_weight + (1 - target) * neg_weight
    loss = loss / jnp.mean(avg_weight)
    return jnp.mean(loss)
