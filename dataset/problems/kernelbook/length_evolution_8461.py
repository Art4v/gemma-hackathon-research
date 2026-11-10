"""Absolute forward differences along H and W, class-weighted, summed into a curve length (length_evolution)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'length_evolution_8461',
    'batch': 8,
    'channels': 128,
    'height': 256,
    'width': 256,
    'weight_height': 1,
    'weight_width': 1,
    'func': 'l1',
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    k_mask, k_weight = jax.random.split(key)
    mask_score = jax.random.uniform(
        k_mask,
        (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width']),
        dtype,
    )
    class_weight = jax.random.uniform(
        k_weight,
        (CONFIG['batch'], CONFIG['channels'], CONFIG['weight_height'], CONFIG['weight_width']),
        dtype,
    )
    return mask_score, class_weight


def workload(mask_score, class_weight):
    gradient_h = jnp.abs(mask_score[:, :, 1:, :] - mask_score[:, :, :-1, :])
    gradient_w = jnp.abs(mask_score[:, :, :, 1:] - mask_score[:, :, :, :-1])
    if CONFIG['func'] == 'l2':
        gradient_h = gradient_h * gradient_h
        gradient_w = gradient_w * gradient_w
    return jnp.sum(class_weight * gradient_h) + jnp.sum(class_weight * gradient_w)
