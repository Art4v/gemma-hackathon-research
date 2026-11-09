"""IID isotropic Gaussian UV loss: softplus(sigma)+lb, squared UV residual, NLL, sum."""
import math

import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'IIDIsotropicGaussianUVLoss_581',
    'shape': (8, 128, 128, 128),
    'sigma_lower_bound': 4.0,
    'log2pi': math.log(2 * math.pi),
}


def create_inputs(dtype=jnp.float32):
    keys = jax.random.split(jax.random.key(0), 5)
    shape = CONFIG['shape']
    u, v, sigma_u, target_u, target_v = (
        jax.random.uniform(k, shape, dtype) for k in keys
    )
    return u, v, sigma_u, target_u, target_v


def workload(u, v, sigma_u, target_u, target_v):
    sigma2 = jax.nn.softplus(sigma_u) + CONFIG['sigma_lower_bound']
    delta_t_delta = (u - target_u) ** 2 + (v - target_v) ** 2
    loss = 0.5 * (CONFIG['log2pi'] + 2 * jnp.log(sigma2) + delta_t_delta / sigma2)
    return loss.sum()
