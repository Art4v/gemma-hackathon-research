"""BPR-max loss: softmax(dim=1) * sigmoid(diag - logit), mean, -log."""
import jax
import jax.numpy as jnp

CONFIG = {'name': 'BPR_max_2229', 'n': 8192}


def create_inputs(dtype=jnp.float32):
    n = CONFIG['n']
    logit = jax.random.uniform(jax.random.key(0), (n, n), dtype)
    return (logit,)


def workload(logit):
    logit_softmax = jax.nn.softmax(logit, axis=1)
    diff = jnp.diagonal(logit)[:, None] - logit
    return -jnp.log(jnp.mean(logit_softmax * jax.nn.sigmoid(diff)))
