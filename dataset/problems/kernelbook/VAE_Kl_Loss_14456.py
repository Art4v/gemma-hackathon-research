"""[bf16 variant] VAE KL loss: mean over batch/spatial of sum over dim 1 of 0.5*(mu^2 + exp(v) - v - 1)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'VAE_Kl_Loss_14456',
    'batch': 16,
    'channels': 128,
    'height': 128,
    'width': 128,
    'reduce_dim': 1,
}


def create_inputs(dtype=jnp.bfloat16):
    k1, k2 = jax.random.split(jax.random.key(0))
    shape = (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width'])
    means = jax.random.uniform(k1, shape, dtype)
    variances = jax.random.uniform(k2, shape, dtype)
    return means, variances


def workload(means, variances):
    term = 0.5 * (means ** 2 + jnp.exp(variances) - variances - 1.0)
    return jnp.mean(jnp.sum(term, axis=CONFIG['reduce_dim']))
