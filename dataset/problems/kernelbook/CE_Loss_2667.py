"""Distillation CE loss: log_softmax(student/T) * softmax(teacher/T), summed over all elements, scaled by -T^2/batch."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'CE_Loss_2667',
    'batch': 16,
    'channels': 512,
    'height': 64,
    'width': 64,
    'temperature': 1.0,
}


def create_inputs(dtype=jnp.float32):
    k1, k2 = jax.random.split(jax.random.key(0))
    shape = (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width'])
    output_batch = jax.random.uniform(k1, shape, dtype)
    teacher_outputs = jax.random.uniform(k2, shape, dtype)
    return output_batch, teacher_outputs


def workload(output_batch, teacher_outputs):
    t = CONFIG['temperature']
    log_p = jax.nn.log_softmax(output_batch / t, axis=1)
    q = jax.nn.softmax(teacher_outputs / t, axis=1)
    return -t * t * jnp.sum(log_p * q) / teacher_outputs.shape[0]
