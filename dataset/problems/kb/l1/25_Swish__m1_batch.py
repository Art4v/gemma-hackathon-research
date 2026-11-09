"""[m1 M1_batch variant of kb/l1/25_Swish] Swish x * sigmoid(x) applied elementwise to a leading batch of 2 independent (2048, 16384) activation matrices."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': '25_Swish_m1_batch',
    'num_batches': 2,
    'batch_size': 2048,
    'dim': 16384,
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    shape = (CONFIG['num_batches'], CONFIG['batch_size'], CONFIG['dim'])
    x = jax.random.uniform(key, shape, dtype=dtype)
    return (x,)


def workload(x):
    return x * jax.nn.sigmoid(x)
