"""[m1 M1_batch variant of kb/l1/31_ELU] ELU where(x > 0, x, alpha * (exp(x) - 1)) applied elementwise to a leading batch of 2 independent (4096, 8192) activation matrices."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': '31_ELU_m1_batch',
    'num_batches': 2,
    'batch_size': 4096,
    'dim': 8192,
    'alpha': 1.0,
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    shape = (CONFIG['num_batches'], CONFIG['batch_size'], CONFIG['dim'])
    x = jax.random.uniform(key, shape, dtype=dtype, minval=0.0, maxval=1.0)
    return (x,)


def workload(x):
    alpha = jnp.asarray(CONFIG['alpha'], dtype=x.dtype)
    return jnp.where(x > 0, x, alpha * jnp.expm1(x))
