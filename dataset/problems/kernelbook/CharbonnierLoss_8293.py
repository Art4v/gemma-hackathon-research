"""[bf16 variant] Charbonnier loss: sum(sqrt((pre - gt)^2 + eps^2)) / batch."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'CharbonnierLoss_8293',
    'batch': 16,
    'channels': 32,
    'height': 256,
    'width': 256,
    'eps': 0.001,
}


def create_inputs(dtype=jnp.bfloat16):
    shape = (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width'])
    k_pre, k_gt = jax.random.split(jax.random.key(0))
    pre = jax.random.uniform(k_pre, shape, dtype)
    gt = jax.random.uniform(k_gt, shape, dtype)
    return pre, gt


def workload(pre, gt):
    eps = CONFIG['eps']
    diff = pre - gt
    return jnp.sum(jnp.sqrt(diff * diff + eps ** 2)) / pre.shape[0]
