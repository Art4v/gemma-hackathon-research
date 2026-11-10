"""ExponentialEnvelope: square -> rational (1-d)(1+d) -> exp -> where(d < 1) cutoff."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'ExponentialEnvelope_1696',
    'batch': 16,
    'dim1': 128,
    'dim2': 128,
    'dim3': 128,
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    shape = (CONFIG['batch'], CONFIG['dim1'], CONFIG['dim2'], CONFIG['dim3'])
    d_scaled = jax.random.uniform(key, shape, dtype)
    return (d_scaled,)


def workload(d_scaled):
    env_val = jnp.exp(-d_scaled ** 2 / ((1 - d_scaled) * (1 + d_scaled)))
    return jnp.where(d_scaled < 1, env_val, jnp.zeros_like(d_scaled))
