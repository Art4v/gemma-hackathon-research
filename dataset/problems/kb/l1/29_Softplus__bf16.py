"""[bf16 variant] Softplus activation (beta=1, threshold=20)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': '29_Softplus',
    'batch_size': 4096,
    'dim': 16384,
    'beta': 1.0,
    'threshold': 20.0,
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    x = jax.random.uniform(key, (CONFIG['batch_size'], CONFIG['dim']), dtype=dtype)
    return (x,)


def workload(x):
    beta = CONFIG['beta']
    threshold = CONFIG['threshold']
    bx = x * beta
    safe = jnp.minimum(bx, threshold)
    soft = jnp.log1p(jnp.exp(safe)) / beta
    return jnp.where(bx > threshold, x, soft)
