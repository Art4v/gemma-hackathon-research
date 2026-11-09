"""[bf16 variant] Swish activation (x * sigmoid(x))."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': '25_Swish',
    'batch_size': 4096,
    'dim': 32768,
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    x = jax.random.uniform(key, (CONFIG['batch_size'], CONFIG['dim']), dtype=dtype)
    return (x,)


def workload(x):
    return x * jax.nn.sigmoid(x)
