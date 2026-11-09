"""[bf16 variant] XSigmoidLoss: e = y - y_prime; mean(2*e / (1 + exp(-e)) - e)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'XSigmoidLoss_5349',
    'shape': (16, 32, 256, 256),
}


def create_inputs(dtype=jnp.bfloat16):
    k1, k2 = jax.random.split(jax.random.key(0))
    shape = CONFIG['shape']
    y_t = jax.random.uniform(k1, shape, dtype)
    y_prime_t = jax.random.uniform(k2, shape, dtype)
    return y_t, y_prime_t


def workload(y_t, y_prime_t):
    ey_t = y_t - y_prime_t
    return jnp.mean(2 * ey_t / (1 + jnp.exp(-ey_t)) - ey_t)
