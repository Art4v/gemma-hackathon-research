"""KL distillation loss: log_softmax(student/T) and softmax(teacher/T)+eps -> batchmean KL divergence * T^2."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'KL_Loss_2669',
    'batch': 16,
    'classes': 128,
    'height': 128,
    'width': 128,
    'class_axis': 1,
    'temperature': 1.0,
    'eps': 1e-7,
}


def create_inputs(dtype=jnp.float32):
    shape = (CONFIG['batch'], CONFIG['classes'], CONFIG['height'], CONFIG['width'])
    key = jax.random.key(0)
    k_output, k_teacher = jax.random.split(key)
    output_batch = jax.random.uniform(k_output, shape, dtype)
    teacher_outputs = jax.random.uniform(k_teacher, shape, dtype)
    return output_batch, teacher_outputs


def workload(output_batch, teacher_outputs):
    temperature = CONFIG['temperature']
    axis = CONFIG['class_axis']
    log_probs = jax.nn.log_softmax(output_batch / temperature, axis=axis)
    targets = jax.nn.softmax(teacher_outputs / temperature, axis=axis) + CONFIG['eps']
    pointwise = targets * jnp.log(targets) - targets * log_probs
    kl = jnp.sum(pointwise) / output_batch.shape[0]
    return temperature * temperature * kl
