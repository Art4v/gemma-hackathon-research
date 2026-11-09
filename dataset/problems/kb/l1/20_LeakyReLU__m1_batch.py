"""[m1 M1_batch variant of kb/l1/20_LeakyReLU] LeakyReLU (negative_slope=0.01) applied elementwise to a leading batch of 4 independent (1024, 16384) activation matrices."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': '20_LeakyReLU_m1_batch',
    'num_batches': 4,
    'batch_size': 1024,
    'dim': 16384,
    'negative_slope': 0.01,
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    shape = (CONFIG['num_batches'], CONFIG['batch_size'], CONFIG['dim'])
    x = jax.random.uniform(key, shape, dtype=dtype)
    return (x,)


def workload(x):
    slope = jnp.asarray(CONFIG['negative_slope'], dtype=x.dtype)
    return jnp.where(x > 0, x, x * slope)
