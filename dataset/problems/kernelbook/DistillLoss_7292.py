"""[bf16 variant] DistillLoss: softmax(x/T) -> KL-div batchmean on probabilities * T^2, soft-label cross-entropy, weighted total."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'DistillLoss_7292',
    'batch': 8,
    'classes': 128,
    'height': 128,
    'width': 128,
    'class_axis': 1,
    'temperature': 4.0,
    'distillation_weight': 4.0,
}


def create_inputs(dtype=jnp.bfloat16):
    shape = (CONFIG['batch'], CONFIG['classes'], CONFIG['height'], CONFIG['width'])
    k_outputs, k_labels, k_teacher = jax.random.split(jax.random.key(0), 3)
    outputs = jax.random.uniform(k_outputs, shape, dtype)
    labels = jax.random.uniform(k_labels, shape, dtype)
    outputs_teacher = jax.random.uniform(k_teacher, shape, dtype)
    return outputs, labels, outputs_teacher


def workload(outputs, labels, outputs_teacher):
    axis = CONFIG['class_axis']
    temperature = CONFIG['temperature']
    weight = CONFIG['distillation_weight']

    if weight > 0:
        # Matches the source: the KL "input" is softmax, not log_softmax.
        kl_input = jax.nn.softmax(outputs / temperature, axis=axis)
        kl_target = jax.nn.softmax(outputs_teacher / temperature, axis=axis)
        safe_target = jnp.where(kl_target > 0, kl_target, jnp.ones_like(kl_target))
        pointwise = kl_target * jnp.log(safe_target) - kl_target * kl_input
        soft_target_loss = (jnp.sum(pointwise) / outputs.shape[0]) * temperature ** 2
    else:
        soft_target_loss = jnp.zeros((), outputs.dtype)

    log_probs = jax.nn.log_softmax(outputs, axis=axis)
    hard_target_loss = jnp.mean(-jnp.sum(log_probs * labels, axis=axis))

    total_loss = soft_target_loss * weight + hard_target_loss * (1 - weight)
    return soft_target_loss, hard_target_loss, total_loss
