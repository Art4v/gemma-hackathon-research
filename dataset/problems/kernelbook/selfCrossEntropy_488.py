"""[bf16 variant] selfCrossEntropy_488: softmax(axis=0) -> clamped binary cross-entropy terms -> sum(axis=1) -> negative mean."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'selfCrossEntropy_488',
    'batch': 8,
    'channels': 128,
    'height': 128,
    'width': 128,
    'softmax_axis': 0,
    'sum_axis': 1,
    'eps': 1e-10,
}


def create_inputs(dtype=jnp.bfloat16):
    shape = (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width'])
    k_output, k_target = jax.random.split(jax.random.key(0))
    output = jax.random.uniform(k_output, shape, dtype)
    target = jax.random.uniform(k_target, shape, dtype)
    return output, target


def workload(output, target):
    eps = CONFIG['eps']
    probs = jax.nn.softmax(output, axis=CONFIG['softmax_axis'])
    log_p = jnp.log(jnp.maximum(probs, eps))
    log_not_p = jnp.log(jnp.maximum(1 - probs, eps))
    per_element = target * log_p + (1 - target) * log_not_p
    return -jnp.mean(jnp.sum(per_element, axis=CONFIG['sum_axis']))
