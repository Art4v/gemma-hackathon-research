"""[bf16 variant] LearnableMax: elementwise max(max(x, max1), max2) with per-channel learnable thresholds."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'LearnableMax_2764',
    'batch': 16,
    'channels': 128,
    'height': 128,
    'width': 128,
}


def create_inputs(dtype=jnp.bfloat16):
    k_x, k_m1, k_m2 = jax.random.split(jax.random.key(0), 3)
    b, c, h, w = CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width']
    x = jax.random.uniform(k_x, (b, c, h, w), dtype)
    max1 = jax.random.normal(k_m1, (1, c, 1, 1), dtype) * 0.02
    max2 = jax.random.normal(k_m2, (1, c, 1, 1), dtype) * 0.02
    return x, max1, max2


def workload(x, max1, max2):
    out = jnp.maximum(x, max1)
    return jnp.maximum(out, max2)
