"""[bf16 variant] KDLoss: log_softmax(student/T) + softmax(teacher/T) -> pointwise KL divergence -> sum over axis 1 -> mean -> scale by T^2."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'KDLoss_source_code_17817',
    'batch': 8,
    'channels': 128,
    'height': 128,
    'classes': 128,
    'temp': 4,
    'reduction': 4,
}


def create_inputs(dtype=jnp.bfloat16):
    shape = (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['classes'])
    key = jax.random.key(0)
    k_teacher, k_student = jax.random.split(key)
    teacher_logits = jax.random.uniform(k_teacher, shape, dtype)
    student_logits = jax.random.uniform(k_student, shape, dtype)
    return teacher_logits, student_logits


def workload(teacher_logits, student_logits):
    temp = CONFIG['temp']
    student_log_probs = jax.nn.log_softmax(student_logits / temp, axis=-1)
    teacher_probs = jax.nn.softmax(teacher_logits / temp, axis=-1)
    positive = teacher_probs > 0
    safe_probs = jnp.where(positive, teacher_probs, jnp.ones_like(teacher_probs))
    xlogx = teacher_probs * jnp.log(safe_probs)
    kl = xlogx - teacher_probs * student_log_probs
    if CONFIG['reduction'] == 'sum':
        kl = jnp.sum(kl)
    else:
        kl = jnp.mean(jnp.sum(kl, axis=1))
    return kl * temp ** 2
