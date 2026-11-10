"""[bf16 variant] AttentionLoss: class-balance alpha -> beta**((1-p)**gamma) and beta**(p**gamma) weights -> weighted binary log loss -> mean."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'AttentionLoss_16330',
    'batch': 8,
    'channels': 128,
    'height': 128,
    'width': 128,
    'beta': 4.0,
    'gamma': 0.5,
    'pred_min': 1e-3,
    'pred_max': 0.999,
}


def create_inputs(dtype=jnp.bfloat16):
    shape = (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width'])
    key = jax.random.key(0)
    k_pred, k_gt = jax.random.split(key)
    pred = jax.random.uniform(
        k_pred, shape, dtype, minval=CONFIG['pred_min'], maxval=CONFIG['pred_max']
    )
    gt = jax.random.uniform(k_gt, shape, dtype)
    return pred, gt


def workload(pred, gt):
    beta = CONFIG['beta']
    gamma = CONFIG['gamma']
    num_pos = jnp.sum(gt)
    num_neg = jnp.sum(1 - gt)
    alpha = num_neg / (num_pos + num_neg)
    edge_beta = jnp.power(beta, jnp.power(1 - pred, gamma))
    bg_beta = jnp.power(beta, jnp.power(pred, gamma))
    loss = -alpha * edge_beta * jnp.log(pred) * gt
    loss = loss - (1 - alpha) * bg_beta * jnp.log(1 - pred) * (1 - gt)
    return jnp.mean(loss)
