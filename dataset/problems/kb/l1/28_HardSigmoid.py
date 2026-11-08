"""HardSigmoid activation: clip(x + 3, 0, 6) / 6."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': '28_HardSigmoid',
    'batch_size': 4096,
    'dim': 16384,
    'offset': 3.0,
    'upper': 6.0,
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    x = jax.random.uniform(key, (CONFIG['batch_size'], CONFIG['dim']), dtype=dtype)
    return (x,)


def workload(x):
    offset = CONFIG['offset']
    upper = CONFIG['upper']
    return jnp.clip(x + offset, 0.0, upper) / upper
