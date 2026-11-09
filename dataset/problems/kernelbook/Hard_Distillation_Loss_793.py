"""Hard distillation loss: 0.5 * soft-target CrossEntropy(student) + 0.5 * soft-target CrossEntropy(teacher)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'Hard_Distillation_Loss_793',
    'batch': 16,
    'channels': 128,
    'height': 128,
    'width': 128,
    'class_axis': 1,
    'weight': 0.5,
}


def create_inputs(dtype=jnp.float32):
    shape = (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width'])
    k1, k2, k3 = jax.random.split(jax.random.key(0), 3)
    teacher_y = jax.random.uniform(k1, shape, dtype)
    student_y = jax.random.uniform(k2, shape, dtype)
    y = jax.random.uniform(k3, shape, dtype)
    return teacher_y, student_y, y


def _soft_ce(logits, target):
    # PyTorch CrossEntropyLoss with probability targets: sum over class axis, mean over the rest.
    logp = jax.nn.log_softmax(logits, axis=CONFIG['class_axis'])
    return jnp.mean(-jnp.sum(target * logp, axis=CONFIG['class_axis']))


def workload(teacher_y, student_y, y):
    w = CONFIG['weight']
    return w * _soft_ce(student_y, y) + w * _soft_ce(teacher_y, y)
