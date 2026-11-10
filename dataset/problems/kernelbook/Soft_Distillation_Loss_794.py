"""[bf16 variant] Soft distillation loss: probabilistic cross-entropy over axis 1 plus temperature-scaled KL divergence (mean reduction, raw inputs)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'Soft_Distillation_Loss_794',
    'batch': 8,
    'channels': 128,
    'height': 128,
    'width': 128,
    'class_axis': 1,
    'lambda_balancing': 4,
    'temperature_min': 1.0,
    'temperature_max': 2.0,
}


def create_inputs(dtype=jnp.bfloat16):
    shape = (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width'])
    k_teacher, k_student, k_y, k_temperature = jax.random.split(jax.random.key(0), 4)
    teacher_y = jax.random.uniform(k_teacher, shape, dtype)
    student_y = jax.random.uniform(k_student, shape, dtype)
    y = jax.random.uniform(k_y, shape, dtype)
    temperature = jax.random.uniform(
        k_temperature,
        shape,
        dtype,
        minval=CONFIG['temperature_min'],
        maxval=CONFIG['temperature_max'],
    )
    return teacher_y, student_y, y, temperature


def workload(teacher_y, student_y, y, temperature):
    axis = CONFIG['class_axis']
    lambda_balancing = CONFIG['lambda_balancing']

    log_probs = jax.nn.log_softmax(student_y, axis=axis)
    ce = jnp.mean(-jnp.sum(y * log_probs, axis=axis))

    kl_input = student_y / temperature
    kl_target = teacher_y / temperature
    target_xlogy = jnp.where(
        kl_target == 0,
        jnp.zeros_like(kl_target),
        kl_target * jnp.log(kl_target),
    )
    kld = jnp.mean(target_xlogy - kl_target * kl_input)

    return (1 - lambda_balancing) * ce + lambda_balancing * temperature ** 2 * kld
