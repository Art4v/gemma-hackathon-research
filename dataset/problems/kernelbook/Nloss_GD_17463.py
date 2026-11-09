"""[bf16 variant] Nloss_GD: Gaussian negative log-likelihood, sum over channels of log(beta^2) and -0.5*(y-x)^2*beta^2, summed and divided by batch size."""
import math

import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'Nloss_GD_17463',
    'batch': 8,
    'channels': 128,
    'height': 128,
    'width': 256,
    'dim': 4,
}


def create_inputs(dtype=jnp.bfloat16):
    shape = (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width'])
    kx, ky, kb = jax.random.split(jax.random.key(0), 3)
    x = jax.random.uniform(kx, shape, dtype)
    y = jax.random.uniform(ky, shape, dtype)
    sq_beta = jax.random.uniform(kb, shape, dtype)
    return x, y, sq_beta


def workload(x, y, sq_beta):
    beta = sq_beta ** 2
    log_det_term = 0.5 * jnp.sum(jnp.log(beta), axis=1)
    norm_term = -0.5 * math.log(2 * math.pi) * CONFIG['dim']
    exponent = jnp.sum(-0.5 * (y - x) ** 2 * beta, axis=1)
    log_p = log_det_term + exponent + norm_term
    return jnp.sum(-log_p) / x.shape[0]
