"""Conditional entropy loss: softmax * log_softmax * weight -> sum(classes) -> mean(batch) -> negate."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'ConditionalEntropyLoss_13104',
    'batch': 16,
    'classes': 128,
    'height': 128,
    'width': 128,
    'class_axis': 1,
    'batch_axis': 0,
    'scale': -1.0,
}


def create_inputs(dtype=jnp.float32):
    shape = (CONFIG['batch'], CONFIG['classes'], CONFIG['height'], CONFIG['width'])
    k_x, k_weight = jax.random.split(jax.random.key(0))
    x = jax.random.uniform(k_x, shape, dtype)
    weight = jax.random.uniform(k_weight, shape, dtype)
    return x, weight


def workload(x, weight):
    axis = CONFIG['class_axis']
    loss = jax.nn.softmax(x, axis=axis) * jax.nn.log_softmax(x, axis=axis) * weight
    loss = jnp.sum(loss, axis=axis)
    return CONFIG['scale'] * jnp.mean(loss, axis=CONFIG['batch_axis'])
