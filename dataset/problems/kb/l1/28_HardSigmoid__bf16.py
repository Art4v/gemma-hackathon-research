"""[bf16 variant] HardSigmoid activation."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': '28_HardSigmoid',
    'batch_size': 256,
    'dim': 393216,
}

def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    x = jax.random.uniform(key, (CONFIG['batch_size'], CONFIG['dim']), dtype=dtype)
    return (x,)

def workload(x):
    return jax.nn.hard_sigmoid(x)
