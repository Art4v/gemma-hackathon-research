"""Focal loss: mean binary_cross_entropy_with_logits -> alpha * (1 - exp(-ce))**gamma * ce."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'FocalLoss_2686',
    'batch': 16,
    'channels': 32,
    'height': 256,
    'width': 256,
    'alpha': 1.0,
    'gamma': 2,
}


def create_inputs(dtype=jnp.float32):
    shape = (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width'])
    kx, ky = jax.random.split(jax.random.key(0))
    x = jax.random.uniform(kx, shape, dtype)
    y = jax.random.uniform(ky, shape, dtype)
    return x, y


def workload(x, y):
    # Same numerically stable form PyTorch uses for BCE with logits, mean reduction.
    per_elem = jnp.maximum(x, 0) - x * y + jnp.log1p(jnp.exp(-jnp.abs(x)))
    ce = jnp.mean(per_elem)
    return CONFIG['alpha'] * (1 - jnp.exp(-ce)) ** CONFIG['gamma'] * ce
