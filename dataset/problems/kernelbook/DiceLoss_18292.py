"""Dice loss: sigmoid(alpha * fake) -> global intersection and union sums with smoothing -> 1 - dice."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'DiceLoss_18292',
    'batch': 8,
    'channels': 128,
    'height': 128,
    'width': 128,
    'alpha': 10,
    'smooth': 1,
}


def create_inputs(dtype=jnp.float32):
    shape = (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width'])
    key = jax.random.key(0)
    k_fake, k_real = jax.random.split(key)
    fake = jax.random.uniform(k_fake, shape, dtype)
    real = jax.random.uniform(k_real, shape, dtype)
    return fake, real


def workload(fake, real):
    alpha = CONFIG['alpha']
    smooth = CONFIG['smooth']
    probs = 1.0 / (1.0 + jnp.exp(-alpha * fake))
    intersection = jnp.sum(probs * real) + smooth
    union = jnp.sum(probs) + jnp.sum(real) + smooth
    dice = (2 * intersection) / union
    return 1.0 - dice
