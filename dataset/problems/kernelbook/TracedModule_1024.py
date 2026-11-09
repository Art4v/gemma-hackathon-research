"""[bf16 variant] Cast to float32 -> sqrt -> divide by 5.0 -> floor."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'TracedModule_1024',
    'batch': 16,
    'channels': 16,
    'height': 512,
    'width': 512,
    'divisor': 5.0,
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    shape = (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width'])
    x = jax.random.uniform(key, shape, dtype=dtype)
    return (x,)


def workload(x):
    x = x.astype(jnp.float32)
    return jnp.floor(jnp.sqrt(x) / CONFIG['divisor'])
