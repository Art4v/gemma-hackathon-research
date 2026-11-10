"""[bf16 variant] HLoss entropy: x * log(x), replace NaN with 0, sum, negate."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'HLoss_4771',
    'batch': 8,
    'channels': 128,
    'height': 256,
    'width': 256,
    'nan_fill': 0.0,
    'scale': -1.0,
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    shape = (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width'])
    x = jax.random.uniform(key, shape, dtype)
    return (x,)


def workload(x):
    b = x * jnp.log(x)
    b = jnp.where(jnp.isnan(b), jnp.full_like(b, CONFIG['nan_fill']), b)
    return CONFIG['scale'] * jnp.sum(b)
