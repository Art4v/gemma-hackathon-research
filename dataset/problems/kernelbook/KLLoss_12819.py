"""[bf16 variant] KLLoss: log(|s2|/|s1|) + (s1^2 + (m1-m2)^2) / (2*s2^2) - 0.5, then mean."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'KLLoss_12819',
    'batch': 16,
    'channels': 64,
    'height': 128,
    'width': 128,
}


def create_inputs(dtype=jnp.bfloat16):
    shape = (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width'])
    k_mu, k_sigma, k_tmu, k_tstd = jax.random.split(jax.random.key(0), 4)
    mu = jax.random.uniform(k_mu, shape, dtype)
    sigma = jax.random.uniform(k_sigma, shape, dtype)
    target_mu = jax.random.uniform(k_tmu, shape, dtype)
    target_std = jax.random.uniform(k_tstd, shape, dtype)
    return mu, sigma, target_mu, target_std


def workload(mu, sigma, target_mu, target_std):
    kl = (
        jnp.log(jnp.abs(sigma) / jnp.abs(target_std))
        + (target_std ** 2 + (target_mu - mu) ** 2) / (2 * sigma ** 2)
        - 0.5
    )
    return jnp.mean(kl)
