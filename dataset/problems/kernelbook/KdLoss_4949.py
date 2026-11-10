"""[bf16 variant] KD loss chain: temperature log_softmax/softmax -> pointwise-mean KL divergence + soft-target cross entropy."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'KdLoss_4949',
    'batch': 8,
    'classes': 128,
    'height': 128,
    'width': 128,
    'class_axis': 1,
    'alpha': 0.9,
    'temperature': 5,
}


def create_inputs(dtype=jnp.bfloat16):
    shape = (CONFIG['batch'], CONFIG['classes'], CONFIG['height'], CONFIG['width'])
    k_outputs, k_teacher, k_labels = jax.random.split(jax.random.key(0), 3)
    outputs = jax.random.uniform(k_outputs, shape, dtype)
    teacher_outputs = jax.random.uniform(k_teacher, shape, dtype)
    labels = jax.random.uniform(k_labels, shape, dtype)
    return outputs, teacher_outputs, labels


def workload(outputs, teacher_outputs, labels):
    axis = CONFIG['class_axis']
    alpha = CONFIG['alpha']
    T = CONFIG['temperature']

    log_p = jax.nn.log_softmax(outputs / T, axis=axis)
    q = jax.nn.softmax(teacher_outputs / T, axis=axis)
    q_log_q = jnp.where(q > 0, q * jnp.log(q), jnp.zeros_like(q))
    kl = jnp.mean(q_log_q - q * log_p)

    ce = jnp.mean(-jnp.sum(labels * jax.nn.log_softmax(outputs, axis=axis), axis=axis))

    return kl * (alpha * T * T) + ce * (1.0 - alpha)
