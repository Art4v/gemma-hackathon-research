"""[bf16 variant] L1 triplet margin loss (abs-diff sum, relu) plus soft-target cross-entropy (log_softmax, mean) with weighted total."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'TripletSoftmaxLoss_17721',
    'batch': 8,
    'channels': 128,
    'height': 128,
    'width': 128,
    'margin': 0.0,
    'lambda_factor': 0.0,
    'distance_axis': 1,
    'class_axis': 1,
}


def create_inputs(dtype=jnp.bfloat16):
    shape = (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width'])
    k_anchor, k_positive, k_negative, k_outputs, k_labels = jax.random.split(jax.random.key(0), 5)
    anchor = jax.random.uniform(k_anchor, shape, dtype)
    positive = jax.random.uniform(k_positive, shape, dtype)
    negative = jax.random.uniform(k_negative, shape, dtype)
    outputs = jax.random.uniform(k_outputs, shape, dtype)
    labels = jax.random.uniform(k_labels, shape, dtype)
    return anchor, positive, negative, outputs, labels


def workload(anchor, positive, negative, outputs, labels):
    distance_axis = CONFIG['distance_axis']
    class_axis = CONFIG['class_axis']
    distance_positive = jnp.sum(jnp.abs(anchor - positive), axis=distance_axis)
    distance_negative = jnp.sum(jnp.abs(anchor - negative), axis=distance_axis)
    losses = jnp.maximum(distance_positive - distance_negative + CONFIG['margin'], 0.0)
    losses_sum = jnp.sum(losses)
    log_probs = jax.nn.log_softmax(outputs, axis=class_axis)
    loss_softmax = jnp.mean(-jnp.sum(labels * log_probs, axis=class_axis))
    loss_total = CONFIG['lambda_factor'] * losses_sum + loss_softmax
    return loss_total, losses_sum, loss_softmax
