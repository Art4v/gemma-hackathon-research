"""[bf16 variant] Elementwise ReLU activation."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': '19_ReLU',
    'batch_size': 4096,
    'dim': 32768,
}

def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    shape = (CONFIG['batch_size'], CONFIG['dim'])
    x = jax.random.normal(key, shape, dtype=dtype)
    return (x,)

def workload(x):
    return jax.nn.relu(x)
