"""JSD loss: -(log 2 + 0.5 * (mean(log d_x) + mean(log(1 - d_y))))."""
import math

import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'JSDLoss_4010',
    'batch': 16,
    'channels': 64,
    'height': 128,
    'width': 128,
    'log2': math.log(2.0),
    # Keep inputs away from 0 and 1 so log stays finite. With 16.7M f32 uniforms, an exact 0 is likely.
    'eps_margin': 1e-3,
}


def create_inputs(dtype=jnp.float32):
    shape = (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width'])
    lo = CONFIG['eps_margin']
    hi = 1.0 - CONFIG['eps_margin']
    kx, ky = jax.random.split(jax.random.key(0))
    d_x = jax.random.uniform(kx, shape, dtype, minval=lo, maxval=hi)
    d_y = jax.random.uniform(ky, shape, dtype, minval=lo, maxval=hi)
    return d_x, d_y


def workload(d_x, d_y):
    real_term = jnp.mean(jnp.log(d_x))
    fake_term = jnp.mean(jnp.log(1.0 - d_y))
    return -(CONFIG['log2'] + 0.5 * (real_term + fake_term))
