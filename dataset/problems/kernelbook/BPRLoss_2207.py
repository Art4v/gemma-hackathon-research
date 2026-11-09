"""BPRLoss: diag(logit)[:, None] - logit -> log_sigmoid -> negative mean."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'BPRLoss_2207',
    'batch': 8192,
}


def create_inputs(dtype=jnp.float32):
    b = CONFIG['batch']
    logit = jax.random.uniform(jax.random.key(0), (b, b), dtype=dtype)
    return (logit,)


def workload(logit):
    diff = jnp.diagonal(logit)[:, None] - logit
    return -jnp.mean(jax.nn.log_sigmoid(diff))
