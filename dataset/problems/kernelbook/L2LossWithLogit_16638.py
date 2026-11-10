"""Sigmoid followed by sum-reduced MSE loss (L2LossWithLogit)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'L2LossWithLogit_16638',
    'batch': 8,
    'channels': 16,
    'height': 512,
    'width': 512,
}


def create_inputs(dtype=jnp.float32):
    shape = (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width'])
    k1, k2 = jax.random.split(jax.random.key(0))
    logits = jax.random.uniform(k1, shape, dtype)
    targets = jax.random.uniform(k2, shape, dtype)
    return logits, targets


def workload(logits, targets):
    p = jax.nn.sigmoid(logits)
    return jnp.sum((p - targets) ** 2)
