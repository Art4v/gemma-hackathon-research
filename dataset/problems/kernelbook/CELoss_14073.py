"""Distillation cross-entropy: per-slice log_softmax(student/T) and softmax(teacher/T) along axis 2, sum(-t*s), mean, sum over slices, scale."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'CELoss_14073',
    'num_preds': 8,
    'batch': 128,
    'classes': 128,
    'spatial': 256,
    'class_axis': 2,
    'T': 1.0,
    'loss_weight': 1.0,
}


def create_inputs(dtype=jnp.float32):
    shape = (CONFIG['num_preds'], CONFIG['batch'], CONFIG['classes'], CONFIG['spatial'])
    key = jax.random.key(0)
    k_s, k_t = jax.random.split(key)
    s_preds = jax.random.uniform(k_s, shape, dtype)
    t_preds = jax.random.uniform(k_t, shape, dtype)
    return s_preds, t_preds


def workload(s_preds, t_preds):
    axis = CONFIG['class_axis']
    temperature = CONFIG['T']
    s = jax.nn.log_softmax(s_preds / temperature, axis=axis)
    t = jax.nn.softmax(t_preds / temperature, axis=axis)
    per_element = jnp.sum(-t * s, axis=axis)
    per_slice = jnp.mean(per_element, axis=tuple(range(1, per_element.ndim)))
    return jnp.sum(per_slice, axis=0) * CONFIG['loss_weight']
