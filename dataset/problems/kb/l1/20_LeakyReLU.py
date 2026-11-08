"""LeakyReLU activation (elementwise, negative_slope=0.01)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': '20_LeakyReLU',
    'batch_size': 4096,
    'dim': 16384,
    'negative_slope': 0.01,
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    x = jax.random.uniform(key, (CONFIG['batch_size'], CONFIG['dim']), dtype=dtype)
    return (x,)


def workload(x):
    slope = jnp.asarray(CONFIG['negative_slope'], dtype=x.dtype)
    return jnp.where(x > 0, x, x * slope)
