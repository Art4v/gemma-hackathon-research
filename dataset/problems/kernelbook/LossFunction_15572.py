"""TOP1 loss: mean(sigmoid(logit - diag(logit)[:, None])) + mean(sigmoid(logit ** 2))."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'LossFunction_15572',
    'batch': 8192,
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    b = CONFIG['batch']
    logit = jax.random.uniform(key, (b, b), dtype=dtype)
    return (logit,)


def workload(logit):
    diff = logit - jnp.diagonal(logit)[:, None]
    return jnp.mean(jax.nn.sigmoid(diff)) + jnp.mean(jax.nn.sigmoid(logit ** 2))
