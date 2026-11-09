"""Sigmoid followed by mean binary cross-entropy (BCE with log clamp at -100)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'BCE_17603',
    'batch': 8,
    'channels': 64,
    'height': 256,
    'width': 256,
    'log_clamp': -100.0,
}


def create_inputs(dtype=jnp.float32):
    shape = (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width'])
    k_logit, k_target = jax.random.split(jax.random.key(0))
    logit = jax.random.uniform(k_logit, shape, dtype)
    target = jax.random.uniform(k_target, shape, dtype)
    return logit, target


def workload(logit, target):
    target = target.astype(jnp.float32)
    pred = jax.nn.sigmoid(logit.astype(jnp.float32))
    clamp = CONFIG['log_clamp']
    log_p = jnp.maximum(jnp.log(pred), clamp)
    log_1mp = jnp.maximum(jnp.log(1.0 - pred), clamp)
    loss = -(target * log_p + (1.0 - target) * log_1mp)
    return jnp.mean(loss)
