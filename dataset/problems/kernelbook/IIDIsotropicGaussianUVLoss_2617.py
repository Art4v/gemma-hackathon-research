"""[bf16 variant] IID isotropic Gaussian UV loss: softplus(sigma)+lb -> 0.5*(log2pi + 2*log(sigma2) + |delta|^2/sigma2) -> sum."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'IIDIsotropicGaussianUVLoss_2617',
    'batch': 8,
    'channels': 32,
    'height': 256,
    'width': 256,
    'sigma_lower_bound': 4.0,
}


def create_inputs(dtype=jnp.bfloat16):
    shape = (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width'])
    keys = jax.random.split(jax.random.key(0), 5)
    u, v, sigma_u, target_u, target_v = (
        jax.random.uniform(k, shape, dtype) for k in keys
    )
    return u, v, sigma_u, target_u, target_v


def workload(u, v, sigma_u, target_u, target_v):
    log2pi = jnp.log(2.0 * jnp.pi)
    sigma2 = jax.nn.softplus(sigma_u) + CONFIG['sigma_lower_bound']
    delta_t_delta = (u - target_u) ** 2 + (v - target_v) ** 2
    loss = 0.5 * (log2pi + 2.0 * jnp.log(sigma2) + delta_t_delta / sigma2)
    return jnp.sum(loss)
