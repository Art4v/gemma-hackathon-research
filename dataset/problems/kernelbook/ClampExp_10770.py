"""[bf16 variant] ClampExp: min(exp(x), 1)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'ClampExp_10770',
    'batch': 16,
    'channels': 32,
    'height': 256,
    'width': 512,
    'clamp_max': 1.0,
}


def create_inputs(dtype=jnp.bfloat16):
    shape = (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width'])
    x = jax.random.uniform(jax.random.key(0), shape, dtype)
    return (x,)


def workload(x):
    return jnp.minimum(jnp.exp(x), jnp.asarray(CONFIG['clamp_max'], dtype=x.dtype))
