"""Concatenate positive and negative logits along the last axis, build 1/0 labels, mean binary cross-entropy with logits."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'MyBCEWithLogitsLoss_15557',
    'batch': 8,
    'dim1': 128,
    'dim2': 128,
    'dim3': 128,
    'concat_axis': -1,
}


def create_inputs(dtype=jnp.float32):
    shape = (CONFIG['batch'], CONFIG['dim1'], CONFIG['dim2'], CONFIG['dim3'])
    key = jax.random.key(0)
    k_pos, k_neg = jax.random.split(key)
    positives = jax.random.uniform(k_pos, shape, dtype)
    negatives = jax.random.uniform(k_neg, shape, dtype)
    return positives, negatives


def workload(positives, negatives):
    axis = CONFIG['concat_axis']
    values = jnp.concatenate((positives, negatives), axis=axis)
    labels = jnp.concatenate(
        (jnp.ones_like(positives), jnp.zeros_like(negatives)), axis=axis
    )
    losses = (1 - labels) * values + jax.nn.softplus(-values)
    return jnp.mean(losses)
