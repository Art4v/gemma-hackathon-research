"""[bf16 variant] Weighted BCE-with-logits loss: pos_weight = neg/pos count ratio, scaled by pos fraction."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'MyWcploss_16494',
    'batch': 16,
    'channels': 32,
    'height': 256,
    'width': 256,
    'eps': 1e-10,
}


def create_inputs(dtype=jnp.bfloat16):
    k_pred, k_gt = jax.random.split(jax.random.key(0))
    shape = (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width'])
    pred = jax.random.normal(k_pred, shape, dtype)
    gt = jax.random.uniform(k_gt, shape, dtype)
    return pred, gt


def workload(pred, gt):
    count_pos = jnp.sum(gt) + CONFIG['eps']
    count_neg = jnp.sum(1.0 - gt)
    beta = count_neg / count_pos
    beta_back = count_pos / (count_pos + count_neg)
    # Same stable formulation as torch's binary_cross_entropy_with_logits with pos_weight.
    log_weight = (beta - 1.0) * gt + 1.0
    softplus_neg = jnp.log1p(jnp.exp(-jnp.abs(pred))) + jnp.maximum(-pred, 0.0)
    bce = (1.0 - gt) * pred + log_weight * softplus_neg
    return beta_back * jnp.mean(bce)
