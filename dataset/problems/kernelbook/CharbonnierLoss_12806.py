"""[bf16 variant] Charbonnier loss: mean(sqrt((x - y)^2 + eps))."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'CharbonnierLoss_12806',
    'batch': 16,
    'channels': 64,
    'height': 256,
    'width': 256,
    'eps': 1e-6,
}


def create_inputs(dtype=jnp.bfloat16):
    kx, ky = jax.random.split(jax.random.key(0))
    shape = (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width'])
    x = jax.random.uniform(kx, shape, dtype)
    y = jax.random.uniform(ky, shape, dtype)
    return x, y


def workload(x, y):
    # mode=None and mask=None in the reference module, so this is a plain mean.
    diff = x - y
    return jnp.mean(jnp.sqrt(diff * diff + CONFIG['eps']))
