"""[bf16 variant] ActNorm forward: z = x * exp(log_sigma) + mu over the last axis, log_det = sum(log_sigma)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'ActNorm_6985',
    'batch': 16,
    'h': 64,
    'w': 128,
    'dim': 512,
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    x = jax.random.uniform(
        key, (CONFIG['batch'], CONFIG['h'], CONFIG['w'], CONFIG['dim']), dtype=dtype
    )
    mu = jnp.zeros((CONFIG['dim'],), dtype=dtype)
    log_sigma = jnp.zeros((CONFIG['dim'],), dtype=dtype)
    return x, mu, log_sigma


def workload(x, mu, log_sigma):
    z = x * jnp.exp(log_sigma) + mu
    log_det = jnp.sum(log_sigma)
    return z, log_det
