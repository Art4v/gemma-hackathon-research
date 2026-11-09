"""DiceLoss: sigmoid -> per-channel dice on batch element 0 (first column dropped) -> mean(1 - dice)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'DiceLoss_5958',
    'batch': 8,
    'channels': 4,
    'height': 1024,
    'width': 1024,
    'epsilon': 1e-5,
}


def create_inputs(dtype=jnp.float32):
    k1, k2 = jax.random.split(jax.random.key(0))
    shape = (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width'])
    x = jax.random.uniform(k1, shape, dtype)
    target = jax.random.uniform(k2, shape, dtype)
    return x, target


def workload(x, target):
    rows = CONFIG['channels']
    # Only batch element 0 reaches the output, matching the original module.
    p = jax.nn.sigmoid(x)[0].reshape(rows, -1)[:, 1:]
    t = target[0].reshape(rows, -1)[:, 1:]
    intersect = jnp.sum(p * t, axis=-1)
    denominator = jnp.sum(p + t, axis=-1)
    dice = 2.0 * intersect / jnp.maximum(denominator, CONFIG['epsilon'])
    return jnp.mean(1.0 - dice)
