"""[bf16 variant] FocalLoss: log_sigmoid / log_minus_sigmoid, sigmoid focal weights (gamma), weighted BCE normalized by mean weight, then mean."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'FocalLoss_15384',
    'batch': 8,
    'channels': 64,
    'height': 256,
    'width': 256,
    'gamma': 2,
}


def create_inputs(dtype=jnp.bfloat16):
    shape = (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width'])
    k_in, k_tgt = jax.random.split(jax.random.key(0))
    x = jax.random.uniform(k_in, shape, dtype)
    target = jax.random.uniform(k_tgt, shape, dtype)
    return x, target


def _log_sigmoid(x):
    return jnp.minimum(x, 0) - jnp.log(1 + jnp.exp(-jnp.abs(x))) + 0.5 * jnp.clip(x, 0, 0)


def _log_minus_sigmoid(x):
    return jnp.minimum(-x, 0) - jnp.log(1 + jnp.exp(-jnp.abs(x))) + 0.5 * jnp.clip(x, 0, 0)


def workload(x, target):
    gamma = CONFIG['gamma']
    pos_log_sig = _log_sigmoid(x)
    neg_log_sig = _log_minus_sigmoid(x)
    prob = jax.nn.sigmoid(x)
    pos_weight = jnp.power(1 - prob, gamma)
    neg_weight = jnp.power(prob, gamma)
    loss = -(target * pos_weight * pos_log_sig + (1 - target) * neg_weight * neg_log_sig)
    avg_weight = target * pos_weight + (1 - target) * neg_weight
    loss = loss / jnp.mean(avg_weight)
    return jnp.mean(loss)
