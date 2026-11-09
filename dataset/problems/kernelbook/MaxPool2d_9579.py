"""[bf16 variant] Probabilistic (moment-matching) 2x2 max pool: 1x2 pool over W, then 2x1 pool over H, on Gaussian mean/variance."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'MaxPool2d_9579',
    'batch': 16,
    'channels': 32,
    'height': 256,
    'width': 256,
}


def create_inputs(dtype=jnp.bfloat16):
    k_mean, k_var = jax.random.split(jax.random.key(0))
    shape = (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width'])
    inputs_mean = jax.random.uniform(k_mean, shape, dtype)
    inputs_variance = jax.random.uniform(k_var, shape, dtype)
    return inputs_mean, inputs_variance


def _max_pool_internal(mu_a, mu_b, var_a, var_b):
    stddev = jnp.sqrt(var_a + var_b)
    ab = mu_a - mu_b
    alpha = ab / stddev
    log_norm = jnp.log(jnp.sqrt(2.0 * jnp.pi)).astype(alpha.dtype)
    pdf = jnp.exp(-(alpha ** 2) / 2.0 - log_norm)
    cdf = 0.5 * (1.0 + lax.erf(alpha / jnp.sqrt(2.0).astype(alpha.dtype)))
    z_mu = stddev * pdf + ab * cdf + mu_b
    z_var = ((mu_a + mu_b) * stddev * pdf
             + (mu_a ** 2 + var_a) * cdf
             + (mu_b ** 2 + var_b) * (1.0 - cdf)
             - z_mu ** 2)
    return z_mu, z_var


def workload(inputs_mean, inputs_variance):
    # 1x2 pool along W
    z_mean, z_var = _max_pool_internal(
        inputs_mean[:, :, :, 0::2], inputs_mean[:, :, :, 1::2],
        inputs_variance[:, :, :, 0::2], inputs_variance[:, :, :, 1::2])
    # 2x1 pool along H
    return _max_pool_internal(
        z_mean[:, :, 0::2, :], z_mean[:, :, 1::2, :],
        z_var[:, :, 0::2, :], z_var[:, :, 1::2, :])
