"""[bf16 variant] Masked MSE loss: mask = target != 0; sum((target*mask - outputs*mask)^2) / sum(mask)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'MSELoss_14849',
    'batch': 16,
    'channels': 32,
    'height': 256,
    'width': 256,
}


def create_inputs(dtype=jnp.bfloat16):
    shape = (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width'])
    k_out, k_tgt = jax.random.split(jax.random.key(0))
    outputs = jax.random.uniform(k_out, shape, dtype)
    target = jax.random.uniform(k_tgt, shape, dtype)
    return outputs, target


def workload(outputs, target):
    val_pixels = (target != 0).astype(outputs.dtype)
    loss = target * val_pixels - outputs * val_pixels
    return jnp.sum(loss ** 2) / jnp.sum(val_pixels)
