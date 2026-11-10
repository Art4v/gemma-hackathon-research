"""LogCoshLoss: abs difference + eps -> cosh -> log -> mean."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'LogCoshLoss_4093',
    'batch': 8,
    'dim1': 128,
    'dim2': 128,
    'dim3': 128,
    'eps': 1e-16,
}


def create_inputs(dtype=jnp.float32):
    shape = (CONFIG['batch'], CONFIG['dim1'], CONFIG['dim2'], CONFIG['dim3'])
    key = jax.random.key(0)
    k_y, k_y_prime = jax.random.split(key)
    y_t = jax.random.uniform(k_y, shape, dtype)
    y_prime_t = jax.random.uniform(k_y_prime, shape, dtype)
    return y_t, y_prime_t


def workload(y_t, y_prime_t):
    ey_t = jnp.abs(y_t - y_prime_t)
    return jnp.mean(jnp.log(jnp.cosh(ey_t + CONFIG['eps'])))
