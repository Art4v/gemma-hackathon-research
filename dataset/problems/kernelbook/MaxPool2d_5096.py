"""[bf16 variant] Gaussian moment-matching MaxPool2d: 1x2 width-pair max of (mean, variance) followed by 2x1 height-pair max using erf-based normal pdf/cdf."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'MaxPool2d_5096',
    'batch': 8,
    'channels': 128,
    'height': 256,
    'width': 256,
    'pool_stride': 2,
    'sqrt_2': 2.0 ** 0.5,
    'log_sqrt_2pi': 0.9189385332046727,
}


def create_inputs(dtype=jnp.bfloat16):
    shape = (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width'])
    key = jax.random.key(0)
    k_mean, k_var = jax.random.split(key)
    inputs_mean = jax.random.uniform(k_mean, shape, dtype)
    inputs_variance = jax.random.uniform(k_var, shape, dtype)
    return inputs_mean, inputs_variance


def _max_pool_internal(mu_a, mu_b, var_a, var_b):
    stddev = jnp.sqrt(var_a + var_b)
    ab = mu_a - mu_b
    alpha = ab / stddev
    pdf = jnp.exp(-(alpha ** 2) / 2.0 - CONFIG['log_sqrt_2pi'])
    cdf = 0.5 * (1.0 + lax.erf(alpha / CONFIG['sqrt_2']))
    z_mu = stddev * pdf + ab * cdf + mu_b
    z_var = (
        (mu_a + mu_b) * stddev * pdf
        + (mu_a ** 2 + var_a) * cdf
        + (mu_b ** 2 + var_b) * (1.0 - cdf)
        - z_mu ** 2
    )
    return z_mu, z_var


def workload(inputs_mean, inputs_variance):
    s = CONFIG['pool_stride']
    z_mean, z_variance = _max_pool_internal(
        inputs_mean[:, :, :, 0::s],
        inputs_mean[:, :, :, 1::s],
        inputs_variance[:, :, :, 0::s],
        inputs_variance[:, :, :, 1::s],
    )
    outputs_mean, outputs_variance = _max_pool_internal(
        z_mean[:, :, 0::s, :],
        z_mean[:, :, 1::s, :],
        z_variance[:, :, 0::s, :],
        z_variance[:, :, 1::s, :],
    )
    return outputs_mean, outputs_variance
