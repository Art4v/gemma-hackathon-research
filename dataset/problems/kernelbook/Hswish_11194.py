"""[bf16 variant] Hardswish: x * relu6(x + 3) / 6."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'Hswish_11194',
    'batch': 16,
    'channels': 64,
    'height': 256,
    'width': 256,
}


def create_inputs(dtype=jnp.bfloat16):
    shape = (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width'])
    x = jax.random.uniform(jax.random.key(0), shape, dtype)
    return (x,)


def workload(x):
    return x * jnp.clip(x + 3.0, 0.0, 6.0) / 6.0
