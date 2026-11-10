"""[bf16 variant] KLDivLoss: -0.5 * mean(1 + sigma_hat - mu**2 - exp(sigma_hat)) against N(0, 1)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'KLDivLoss_9895',
    'batch': 32,
    'channels': 64,
    'height': 128,
    'width': 128,
    'scale': -0.5,
}


def create_inputs(dtype=jnp.bfloat16):
    shape = (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width'])
    k1, k2 = jax.random.split(jax.random.key(0))
    sigma_hat = jax.random.uniform(k1, shape, dtype)
    mu = jax.random.uniform(k2, shape, dtype)
    return sigma_hat, mu


def workload(sigma_hat, mu):
    return CONFIG['scale'] * jnp.mean(1 + sigma_hat - mu ** 2 - jnp.exp(sigma_hat))
