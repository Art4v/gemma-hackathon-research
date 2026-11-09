"""[bf16 variant] GaborConstraint: clamp mu channel to [0, pi] and sigma channel to FWHM bounds, stack on dim 1."""
import math

import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'GaborConstraint_15318',
    'batch': 8,
    'channels': 4,
    'height': 1024,
    'width': 1024,
    'kernel_size': 4,
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    shape = (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width'])
    kernel = jax.random.uniform(key, shape, dtype)
    return (kernel,)


def workload(kernel):
    fwhm = math.sqrt(2 * math.log(2)) / math.pi
    mu_lower, mu_upper = 0.0, math.pi
    sigma_lower = 4 * fwhm
    sigma_upper = CONFIG['kernel_size'] * fwhm
    clipped_mu = jnp.clip(kernel[:, 0], mu_lower, mu_upper)
    clipped_sigma = jnp.clip(kernel[:, 1], sigma_lower, sigma_upper)
    return jnp.stack([clipped_mu, clipped_sigma], axis=1)
