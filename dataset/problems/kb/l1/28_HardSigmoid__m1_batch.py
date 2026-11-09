"""[m1 M1_batch variant of kb/l1/28_HardSigmoid] HardSigmoid clip(x + 3, 0, 6) / 6 applied elementwise to a leading batch of 8 independent (512, 16384) activation matrices."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': '28_HardSigmoid_m1_batch',
    'num_batches': 8,
    'batch_size': 512,
    'dim': 16384,
    'offset': 3.0,
    'upper': 6.0,
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    shape = (CONFIG['num_batches'], CONFIG['batch_size'], CONFIG['dim'])
    x = jax.random.uniform(key, shape, dtype=dtype)
    return (x,)


def workload(x):
    offset = CONFIG['offset']
    upper = CONFIG['upper']
    return jnp.clip(x + offset, 0.0, upper) / upper
