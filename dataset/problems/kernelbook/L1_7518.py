"""L1 (Charbonnier) loss: sqrt((x - target)^2 + eps), sum over (C, H, W), mean over batch."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'L1_7518',
    'batch': 16,
    'channels': 32,
    'height': 256,
    'width': 256,
    'eps': 1e-6,
}


def create_inputs(dtype=jnp.float32):
    k1, k2 = jax.random.split(jax.random.key(0))
    shape = (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width'])
    x = jax.random.uniform(k1, shape, dtype)
    target = jax.random.uniform(k2, shape, dtype)
    return x, target


def workload(x, target):
    diff = x - target
    per_sample = jnp.sum(jnp.sqrt(diff * diff + CONFIG['eps']), axis=(1, 2, 3))
    return jnp.mean(per_sample)
