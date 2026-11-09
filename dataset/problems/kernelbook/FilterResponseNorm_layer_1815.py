"""[bf16 variant] FilterResponseNorm: x * rsqrt(mean(x^2, HW) + eps), affine (gamma, beta), then max with tau (TLU)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'FilterResponseNorm_layer_1815',
    'batch': 16,
    'channels': 128,
    'height': 128,
    'width': 128,
    'eps': 1e-6,
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    n, c, h, w = CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width']
    x = jax.random.uniform(key, (n, c, h, w), dtype)
    par_shape = (1, c, 1, 1)
    gamma = jnp.ones(par_shape, dtype)
    beta = jnp.zeros(par_shape, dtype)
    tau = jnp.zeros(par_shape, dtype)
    return x, gamma, beta, tau


def workload(x, gamma, beta, tau):
    nu2 = jnp.mean(jnp.square(x), axis=(2, 3), keepdims=True)
    x = x * (1.0 / jnp.sqrt(nu2 + CONFIG['eps']))
    y = gamma * x + beta
    return jnp.maximum(y, tau)
