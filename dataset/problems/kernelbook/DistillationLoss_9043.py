"""DistillationLoss: exp(log_softmax(teacher/T)) * -log_softmax(student/T), sum over last dim, mean."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'DistillationLoss_9043',
    'dim0': 16,
    'dim1': 16,
    'dim2': 128,
    'dim3': 1024,
    'temperature': 1.0,
}


def create_inputs(dtype=jnp.float32):
    k1, k2 = jax.random.split(jax.random.key(0))
    shape = (CONFIG['dim0'], CONFIG['dim1'], CONFIG['dim2'], CONFIG['dim3'])
    student_logits = jax.random.uniform(k1, shape, dtype)
    teacher_logits = jax.random.uniform(k2, shape, dtype)
    return student_logits, teacher_logits


def workload(student_logits, teacher_logits):
    t = CONFIG['temperature']
    teacher_prediction = jnp.exp(jax.nn.log_softmax(teacher_logits / t, axis=-1))
    student_prediction = jax.nn.log_softmax(student_logits / t, axis=-1)
    return jnp.mean(jnp.sum(-teacher_prediction * student_prediction, axis=-1))
