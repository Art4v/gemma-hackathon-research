"""[bf16 variant] DistilledLoss: soft-target cross_entropy(dim=1) + T^2 * batchmean kl_div(log_softmax, softmax over last dim), alpha-blended."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'DistilledLoss_17311',
    'batch': 16,
    'channels': 128,
    'height': 64,
    'width': 128,
    'alpha': 0.5,
    'temperature': 1.0,
}


def create_inputs(dtype=jnp.bfloat16):
    shape = (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width'])
    k1, k2, k3, k4 = jax.random.split(jax.random.key(0), 4)
    teacher_logits = jax.random.uniform(k1, shape, dtype)
    student_logits = jax.random.uniform(k2, shape, dtype)
    distill_logits = jax.random.uniform(k3, shape, dtype)
    labels = jax.random.uniform(k4, shape, dtype)
    return teacher_logits, student_logits, distill_logits, labels


def workload(teacher_logits, student_logits, distill_logits, labels):
    alpha = CONFIG['alpha']
    t = CONFIG['temperature']
    # Soft-label cross entropy: class axis is 1, mean over all non-class positions.
    ce = jnp.mean(-jnp.sum(labels * jax.nn.log_softmax(student_logits, axis=1), axis=1))
    # kl_div(input=log q, target=p): xlogy(p, p) - p * log q, summed, divided by batch size.
    log_q = jax.nn.log_softmax(distill_logits / t, axis=-1)
    p = jax.lax.stop_gradient(jax.nn.softmax(teacher_logits / t, axis=-1))
    kl = jnp.sum(jax.scipy.special.xlogy(p, p) - p * log_q) / distill_logits.shape[0]
    kl = kl * t ** 2
    return ce * alpha + kl * (1 - alpha)
