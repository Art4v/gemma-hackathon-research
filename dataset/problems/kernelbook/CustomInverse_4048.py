"""[bf16 variant] Batched matrix inverse plus residual (inv(x) + x) and all-nonzero reduction of y."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'CustomInverse_4048',
    'batch': 32,
    'channels': 16,
    'n': 128,
}


def create_inputs(dtype=jnp.bfloat16):
    k1, k2 = jax.random.split(jax.random.key(0))
    shape = (CONFIG['batch'], CONFIG['channels'], CONFIG['n'], CONFIG['n'])
    x = jax.random.uniform(k1, shape, dtype)
    y = jax.random.uniform(k2, shape, dtype)
    return x, y


def workload(x, y):
    # Inverse over the trailing (n, n) matrix dims, batched over leading dims.
    ress = jnp.linalg.inv(x) + x
    # torch.all on floats: True iff every element is nonzero.
    return ress, jnp.all(y)
