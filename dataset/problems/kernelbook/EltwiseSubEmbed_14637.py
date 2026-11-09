"""[bf16 variant] EltwiseSubEmbed: (x1 - x2)^2 summed over the channel dim."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'EltwiseSubEmbed_14637',
    'batch': 16,
    'channels': 256,
    'height': 64,
    'width': 128,
    'reduce_axis': 1,
}


def create_inputs(dtype=jnp.bfloat16):
    k1, k2 = jax.random.split(jax.random.key(0))
    shape = (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width'])
    x1 = jax.random.uniform(k1, shape, dtype)
    x2 = jax.random.uniform(k2, shape, dtype)
    return x1, x2


def workload(x1, x2):
    d = x1 - x2
    return jnp.sum(d * d, axis=CONFIG['reduce_axis'])
