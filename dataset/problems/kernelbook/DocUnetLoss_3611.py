"""[bf16 variant] DocUnetLoss: mse(y, label) + mean|y - label| - r * |mean(y - label)|."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'DocUnetLoss_3611',
    'batch': 8,
    'channels': 16,
    'height': 512,
    'width': 512,
    'r': 0.1,
}


def create_inputs(dtype=jnp.bfloat16):
    k1, k2 = jax.random.split(jax.random.key(0))
    shape = (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width'])
    y = jax.random.uniform(k1, shape, dtype)
    label = jax.random.uniform(k2, shape, dtype)
    return y, label


def workload(y, label):
    d = y - label
    lossf = jnp.mean(jnp.abs(d)) - CONFIG['r'] * jnp.abs(jnp.mean(d))
    return jnp.mean(d * d) + lossf
