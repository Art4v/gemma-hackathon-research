"""[bf16 variant] SuperLoss (default): weighted MSE loss, w * mean((output - target)^2)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'SuperLoss_6372',
    'batch': 8,
    'channels': 32,
    'height': 256,
    'width': 512,
    'mse_weight': 1.0,
}


def create_inputs(dtype=jnp.bfloat16):
    shape = (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width'])
    k_out, k_tgt = jax.random.split(jax.random.key(0))
    output = jax.random.uniform(k_out, shape, dtype)
    target = jax.random.uniform(k_tgt, shape, dtype)
    return output, target


def workload(output, target):
    return CONFIG['mse_weight'] * jnp.mean((output - target) ** 2)
