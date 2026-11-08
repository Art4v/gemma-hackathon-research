"""ELU activation: where(x > 0, x, alpha * (exp(x) - 1))."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': '31_ELU',
    'batch_size': 4096,
    'dim': 16384,
    'alpha': 1.0,
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    x = jax.random.uniform(key, (CONFIG['batch_size'], CONFIG['dim']), dtype=dtype,
                           minval=0.0, maxval=1.0)
    return (x,)


def workload(x):
    alpha = jnp.asarray(CONFIG['alpha'], dtype=x.dtype)
    return jnp.where(x > 0, x, alpha * jnp.expm1(x))
