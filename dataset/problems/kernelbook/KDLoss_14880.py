"""[bf16 variant] KD loss: log_softmax(student/T), softmax(teacher/T) -> KLDiv(none) -> sum(dim 1).mean() * T^2."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'KDLoss_14880',
    'batch': 8,
    'dim1': 128,
    'dim2': 128,
    'classes': 256,
    'temp': 4.0,
    # Original module passes reduction=4; anything other than 'sum' takes the sum(1).mean() path.
    'reduction': 4,
}


def create_inputs(dtype=jnp.bfloat16):
    shape = (CONFIG['batch'], CONFIG['dim1'], CONFIG['dim2'], CONFIG['classes'])
    k_t, k_s = jax.random.split(jax.random.key(0))
    teacher_logits = jax.random.uniform(k_t, shape, dtype)
    student_logits = jax.random.uniform(k_s, shape, dtype)
    return teacher_logits, student_logits


def workload(teacher_logits, student_logits):
    temp = CONFIG['temp']
    student_log = jax.nn.log_softmax(student_logits / temp, axis=-1)
    teacher_log = jax.nn.log_softmax(teacher_logits / temp, axis=-1)
    teacher_prob = jnp.exp(teacher_log)
    # KLDivLoss(reduction='none'): target * (log(target) - input); zero target gives 0.
    kl = teacher_prob * (teacher_log - student_log)
    if CONFIG['reduction'] == 'sum':
        kl = jnp.sum(kl)
    else:
        kl = jnp.mean(jnp.sum(kl, axis=1))
    return kl * temp ** 2
