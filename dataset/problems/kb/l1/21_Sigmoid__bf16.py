"""[bf16 variant] Element-wise Sigmoid activation."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': '21_Sigmoid',
    'batch_size': 512,
    'dim': 262144,
}

def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    x = jax.random.uniform(key, (CONFIG['batch_size'], CONFIG['dim']), dtype=dtype)
    return (x,)

def workload(x):
    return jax.nn.sigmoid(x)
