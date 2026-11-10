"""PSNR loss: MSE(output, target) -> 10 * log10(1 / mse)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'PSNRLoss_7356',
    'batch': 8,
    'channels': 4,
    'height': 1024,
    'width': 1024,
    'max_val': 1.0,
    'scale': 10.0,
}


def create_inputs(dtype=jnp.float32):
    k1, k2 = jax.random.split(jax.random.key(0))
    shape = (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width'])
    output = jax.random.uniform(k1, shape, dtype)
    target = jax.random.uniform(k2, shape, dtype)
    return output, target


def workload(output, target):
    mse = jnp.mean(jnp.square(output - target))
    return CONFIG['scale'] * jnp.log10(CONFIG['max_val'] / mse)
