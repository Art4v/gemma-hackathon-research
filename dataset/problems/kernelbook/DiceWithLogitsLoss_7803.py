"""Dice loss with logits: sigmoid(pred) -> sum(pred*target), sum(pred), sum(target) -> 1 - dice."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'DiceWithLogitsLoss_7803',
    'batch': 32,
    'channels': 4,
    'height': 512,
    'width': 512,
    'smooth': 1.0,
}


def create_inputs(dtype=jnp.float32):
    k1, k2 = jax.random.split(jax.random.key(0))
    shape = (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width'])
    pred = jax.random.uniform(k1, shape, dtype)
    target = jax.random.uniform(k2, shape, dtype)
    return pred, target


def workload(pred, target):
    smooth = CONFIG['smooth']
    p = jax.nn.sigmoid(pred)
    inter = jnp.sum(p * target)
    z = jnp.sum(p) + jnp.sum(target) + smooth
    return 1.0 - (2.0 * inter + smooth) / z
