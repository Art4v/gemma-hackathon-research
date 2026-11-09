"""Cast to float32 -> sqrt -> divide by 5.0 -> floor (TracedModule_5419)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'TracedModule_5419',
    'batch': 16,
    'channels': 64,
    'height': 256,
    'width': 256,
    'divisor': 5.0,
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    shape = (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width'])
    x = jax.random.uniform(key, shape, dtype=dtype)
    return (x,)


def workload(x):
    x = x.astype(jnp.float32)
    return jnp.floor(jnp.sqrt(x) / CONFIG['divisor'])
