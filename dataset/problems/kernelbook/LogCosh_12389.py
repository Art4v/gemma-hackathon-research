"""LogCosh loss: mean(log(cosh(target - output + bias)))."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'LogCosh_12389',
    'batch': 16,
    'channels': 128,
    'height': 128,
    'width': 128,
    'bias': 1e-12,
}


def create_inputs(dtype=jnp.float32):
    shape = (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width'])
    k_out, k_tgt = jax.random.split(jax.random.key(0))
    output = jax.random.uniform(k_out, shape, dtype)
    target = jax.random.uniform(k_tgt, shape, dtype)
    return output, target


def workload(output, target):
    return jnp.mean(jnp.log(jnp.cosh(target - output + CONFIG['bias'])))
