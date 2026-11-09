"""[bf16 variant] GDiceLossV2: generalized Dice loss with per-channel inverse-squared-volume weights."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'GDiceLossV2_6295',
    'batch': 16,
    'channels': 128,
    'height': 128,
    'width': 128,
    'smooth': 1e-5,
}


def create_inputs(dtype=jnp.bfloat16):
    k1, k2 = jax.random.split(jax.random.key(0))
    shape = (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width'])
    net_output = jax.random.uniform(k1, shape, dtype)
    gt = jax.random.uniform(k2, shape, dtype)
    return net_output, gt


def workload(net_output, gt):
    smooth = CONFIG['smooth']
    x = net_output.astype(jnp.float32)
    t = gt.astype(jnp.float32)
    # Reduce every axis except channel (axis 1), same as PyTorch's flatten to (C, -1).
    axes = (0,) + tuple(range(2, x.ndim))
    target_sum = jnp.sum(t, axis=axes)
    class_weights = 1.0 / jnp.maximum(target_sum * target_sum, smooth)
    intersect = jnp.sum(jnp.sum(x * t, axis=axes) * class_weights)
    denominator = jnp.sum(jnp.sum(x + t, axis=axes) * class_weights)
    return -2.0 * intersect / jnp.maximum(denominator, smooth)
