"""LogLoss: sqrt(MSE(atan(input), atan(target)))."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'LogLoss_4355',
    'batch': 16,
    'channels': 32,
    'height': 256,
    'width': 256,
}


def create_inputs(dtype=jnp.float32):
    k1, k2 = jax.random.split(jax.random.key(0))
    shape = (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width'])
    inp = jax.random.uniform(k1, shape, dtype)
    tgt = jax.random.uniform(k2, shape, dtype)
    return inp, tgt


def workload(inp, tgt):
    diff = jnp.arctan(inp) - jnp.arctan(tgt)
    return jnp.sqrt(jnp.mean(diff * diff))
