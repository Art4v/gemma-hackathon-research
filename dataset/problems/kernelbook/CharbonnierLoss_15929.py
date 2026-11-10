"""[bf16 variant] Charbonnier loss: subtract, square, add eps, sqrt, sum, divide by batch size."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'CharbonnierLoss_15929',
    'batch': 16,
    'channels': 128,
    'height': 128,
    'width': 128,
    'eps': 1e-06,
}


def create_inputs(dtype=jnp.bfloat16):
    shape = (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width'])
    key_pred, key_target = jax.random.split(jax.random.key(0))
    pred = jax.random.uniform(key_pred, shape, dtype)
    target = jax.random.uniform(key_target, shape, dtype)
    return pred, target


def workload(pred, target):
    return jnp.sum(jnp.sqrt((pred - target) ** 2 + CONFIG['eps'])) / target.shape[0]
