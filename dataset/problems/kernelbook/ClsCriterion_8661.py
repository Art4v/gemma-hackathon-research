"""[bf16 variant] ClsCriterion_8661: element-wise product, sum over axis 1, mean, negate (cross-entropy on log-probabilities)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'ClsCriterion_8661',
    'batch': 8,
    'classes': 128,
    'height': 128,
    'width': 128,
    'sum_axis': 1,
}


def create_inputs(dtype=jnp.bfloat16):
    shape = (CONFIG['batch'], CONFIG['classes'], CONFIG['height'], CONFIG['width'])
    key = jax.random.key(0)
    k_predict, k_label = jax.random.split(key)
    predict = jax.random.uniform(k_predict, shape, dtype)
    label = jax.random.uniform(k_label, shape, dtype)
    return predict, label


def workload(predict, label):
    return -jnp.mean(jnp.sum(predict * label, axis=CONFIG['sum_axis']))
