"""[bf16 variant] Triplet margin loss: squared L2 distances over axis 1, relu hinge, mean, plus correct-triplet count."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'tripletLoss_9326',
    'batch': 8,
    'channels': 128,
    'height': 128,
    'width': 128,
    'margin': 4,
    'sum_axis': 1,
}


def create_inputs(dtype=jnp.bfloat16):
    shape = (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width'])
    k_anchor, k_pos, k_neg = jax.random.split(jax.random.key(0), 3)
    anchor = jax.random.uniform(k_anchor, shape, dtype)
    pos = jax.random.uniform(k_pos, shape, dtype)
    neg = jax.random.uniform(k_neg, shape, dtype)
    return anchor, pos, neg


def workload(anchor, pos, neg):
    axis = CONFIG['sum_axis']
    distance_pos = jnp.sum(jnp.square(anchor - pos), axis=axis)
    distance_neg = jnp.sum(jnp.square(anchor - neg), axis=axis)
    loss = jax.nn.relu(distance_pos - distance_neg + CONFIG['margin'])
    correct = jnp.sum(distance_pos < distance_neg)
    return jnp.mean(loss), correct
