"""[bf16 variant] AlgebraicLoss: diff -> e^2 / sqrt(1 + e^2) -> global mean."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'AlgebraicLoss_16630',
    'batch': 8,
    'channels': 64,
    'height': 256,
    'width': 256,
}


def create_inputs(dtype=jnp.bfloat16):
    shape = (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width'])
    k1, k2 = jax.random.split(jax.random.key(0))
    y_t = jax.random.uniform(k1, shape, dtype)
    y_prime_t = jax.random.uniform(k2, shape, dtype)
    return y_t, y_prime_t


def workload(y_t, y_prime_t):
    e = y_t - y_prime_t
    e2 = e * e
    return jnp.mean(e2 / jnp.sqrt(1 + e2))
