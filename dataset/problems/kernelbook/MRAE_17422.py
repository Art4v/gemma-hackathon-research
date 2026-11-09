"""MRAE: mean(|output - target| / (target + 1/65535))."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'MRAE_17422',
    'batch': 8,
    'channels': 64,
    'height': 256,
    'width': 256,
    'eps': 1.0 / 65535.0,
}


def create_inputs(dtype=jnp.float32):
    shape = (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width'])
    k_out, k_tgt = jax.random.split(jax.random.key(0))
    output = jax.random.uniform(k_out, shape, dtype)
    target = jax.random.uniform(k_tgt, shape, dtype)
    return output, target


def workload(output, target):
    relative_diff = jnp.abs(output - target) / (target + CONFIG['eps'])
    return jnp.mean(relative_diff)
