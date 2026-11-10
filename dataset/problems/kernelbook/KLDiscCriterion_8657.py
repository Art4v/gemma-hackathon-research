"""[bf16 variant] KL divergence between discrete distributions: log(gt + eps), exp(log_pre) * (log_pre - log_gt), sum, divide by batch."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'KLDiscCriterion_8657',
    'batch': 16,
    'dim1': 128,
    'dim2': 128,
    'dim3': 128,
    'eps': 0.0001,
    'qp_order': True,
}


def create_inputs(dtype=jnp.bfloat16):
    shape = (CONFIG['batch'], CONFIG['dim1'], CONFIG['dim2'], CONFIG['dim3'])
    k_pre, k_gt = jax.random.split(jax.random.key(0))
    disc_log_pre = jax.random.uniform(k_pre, shape, dtype)
    disc_gt = jax.random.uniform(k_gt, shape, dtype)
    return disc_log_pre, disc_gt


def workload(disc_log_pre, disc_gt):
    batch_size = disc_log_pre.shape[0]
    disc_log_gt = jnp.log(disc_gt + CONFIG['eps'])
    if CONFIG['qp_order']:
        loss = jnp.sum(jnp.exp(disc_log_pre) * (disc_log_pre - disc_log_gt)) / batch_size
    else:
        loss = jnp.sum(disc_gt * (disc_log_gt - disc_log_pre)) / batch_size
    return loss
