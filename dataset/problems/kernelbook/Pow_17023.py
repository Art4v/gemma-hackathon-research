"""[bf16 variant] Pow_17023: elementwise sqrt (x ** sigmoid(a) with default a=0, non-trainable, reduces to torch.sqrt)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'Pow_17023',
    'batch': 8,
    'channels': 128,
    'height': 256,
    'width': 256,
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    shape = (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width'])
    x = jax.random.uniform(key, shape, dtype)
    return (x,)


def workload(x):
    return jnp.sqrt(x)
