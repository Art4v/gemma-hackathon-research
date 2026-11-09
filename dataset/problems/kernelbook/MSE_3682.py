"""[bf16 variant] MSE loss: mean((pred - gt) ** 2) over all elements."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'MSE_3682',
    'batch': 8,
    'channels': 64,
    'height': 256,
    'width': 256,
}


def create_inputs(dtype=jnp.bfloat16):
    shape = (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width'])
    k_pred, k_gt = jax.random.split(jax.random.key(0))
    pred = jax.random.uniform(k_pred, shape, dtype)
    gt = jax.random.uniform(k_gt, shape, dtype)
    return pred, gt


def workload(pred, gt):
    diff = pred - gt
    return jnp.mean(diff * diff)
