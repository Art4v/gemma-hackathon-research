"""SoftEntropy: log_softmax(inputs, class axis) and detached softmax(targets, class axis), cross-entropy mean over batch axis then sum."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'SoftEntropy_2598',
    'batch': 8,
    'classes': 256,
    'height': 128,
    'width': 128,
    'batch_axis': 0,
    'class_axis': 1,
}


def create_inputs(dtype=jnp.float32):
    shape = (CONFIG['batch'], CONFIG['classes'], CONFIG['height'], CONFIG['width'])
    k_inputs, k_targets = jax.random.split(jax.random.key(0))
    inputs = jax.random.uniform(k_inputs, shape, dtype)
    targets = jax.random.uniform(k_targets, shape, dtype)
    return inputs, targets


def workload(inputs, targets):
    log_probs = jax.nn.log_softmax(inputs, axis=CONFIG['class_axis'])
    probs = jax.lax.stop_gradient(jax.nn.softmax(targets, axis=CONFIG['class_axis']))
    return jnp.sum(jnp.mean(-probs * log_probs, axis=CONFIG['batch_axis']))
