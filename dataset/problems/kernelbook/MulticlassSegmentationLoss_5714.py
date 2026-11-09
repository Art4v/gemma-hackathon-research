"""Per-class MSE loss (split by channel, MSE mean per channel, summed over classes)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'MulticlassSegmentationLoss_5714',
    'batch': 16,
    'channels': 8,
    'height': 512,
    'width': 512,
}


def create_inputs(dtype=jnp.float32):
    k1, k2 = jax.random.split(jax.random.key(0))
    shape = (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width'])
    output = jax.random.uniform(k1, shape, dtype)
    target = jax.random.uniform(k2, shape, dtype)
    return output, target


def workload(output, target):
    diff = output - target
    per_class = jnp.mean(diff * diff, axis=(0, 2, 3))
    return jnp.sum(per_class).reshape(1)
