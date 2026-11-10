"""[bf16 variant] Masked sequence NLL: flatten, gather class log-probs, reshape, apply mask, negative masked mean over time."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'StandardNLL_14131',
    'batch': 128,
    'time': 512,
    'num_classes': 512,
}


def create_inputs(dtype=jnp.bfloat16):
    k_log_prob, k_y_true, k_mask = jax.random.split(jax.random.key(0), 3)
    batch, time, num_classes = CONFIG['batch'], CONFIG['time'], CONFIG['num_classes']
    log_prob = -jax.random.uniform(k_log_prob, (batch, time, num_classes), dtype)
    y_true = jax.random.randint(k_y_true, (batch, time), 0, num_classes, jnp.int32)
    mask = jax.random.uniform(k_mask, (batch, time), dtype)
    return log_prob, y_true, mask


def workload(log_prob, y_true, mask):
    mask = mask.astype(log_prob.dtype)
    flat_log_prob = log_prob.reshape(-1, log_prob.shape[2])
    log_p = jnp.take_along_axis(flat_log_prob, y_true.reshape(-1, 1), axis=1)
    log_p = log_p.reshape(y_true.shape[0], y_true.shape[1])
    log_p = log_p * mask
    sum_log_p = jnp.sum(log_p, axis=1) / jnp.sum(mask, axis=1)
    return -sum_log_p
