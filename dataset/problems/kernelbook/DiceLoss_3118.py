"""Dice loss: sigmoid -> element-wise product -> global sums -> 1 - scale * intersection / (sum_pred + sum_target + eps)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'DiceLoss_3118',
    'batch': 8,
    'channels': 128,
    'height': 128,
    'width': 128,
    'eps': 1e-07,
    'dice_scale': 2.0,
}


def create_inputs(dtype=jnp.float32):
    shape = (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width'])
    k_output, k_target = jax.random.split(jax.random.key(0))
    output = jax.random.uniform(k_output, shape, dtype)
    target = jax.random.uniform(k_target, shape, dtype)
    return output, target


def workload(output, target):
    prediction = jax.nn.sigmoid(output)
    numerator = CONFIG['dice_scale'] * jnp.sum(prediction * target)
    denominator = jnp.sum(prediction) + jnp.sum(target) + CONFIG['eps']
    return 1 - numerator / denominator
