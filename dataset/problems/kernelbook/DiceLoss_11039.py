"""DiceLoss: sigmoid -> per-channel dice (sum over N,H,W) -> mean(1 - dice)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'DiceLoss_11039',
    'batch': 16,
    'channels': 32,
    'height': 256,
    'width': 256,
    'epsilon': 1e-5,
}


def create_inputs(dtype=jnp.float32):
    k1, k2 = jax.random.split(jax.random.key(0))
    shape = (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width'])
    x = jax.random.uniform(k1, shape, dtype)
    target = jax.random.uniform(k2, shape, dtype)
    return x, target


def workload(x, target):
    p = jax.nn.sigmoid(x)
    t = target.astype(p.dtype)
    # Torch flattens to (C, N*H*W) and sums the last axis: same as reducing every axis but channel.
    axes = (0,) + tuple(range(2, x.ndim))
    intersect = jnp.sum(p * t, axis=axes)
    denominator = jnp.sum(p + t, axis=axes)
    dice = 2.0 * intersect / jnp.maximum(denominator, CONFIG['epsilon'])
    return jnp.mean(1.0 - dice)
