"""[bf16 variant] CReLU: leaky_relu(x) concatenated with leaky_relu(-leaky_relu(x)) along channels."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'CReLU_15179',
    'batch': 16,
    'channels': 128,
    'height': 128,
    'width': 256,
    'negative_slope': 0.01,
    'concat_axis': 1,
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    shape = (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width'])
    x = jax.random.uniform(key, shape, dtype)
    return (x,)


def workload(x):
    slope = CONFIG['negative_slope']
    # PyTorch's first leaky_relu is inplace, so the negated branch sees the activated x.
    y = jnp.where(x >= 0, x, x * slope)
    neg = -y
    z = jnp.where(neg >= 0, neg, neg * slope)
    return jnp.concatenate([y, z], axis=CONFIG['concat_axis'])
