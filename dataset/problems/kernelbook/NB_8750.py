"""[bf16 variant] Negative binomial log-likelihood (log / lgamma elementwise chain) followed by mean reduction."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'NB_8750',
    'shape': (16, 32, 256, 256),
    'eps': 1e-08,
    'reduction': 'mean',
}


def create_inputs(dtype=jnp.bfloat16):
    kx, kmu, kth = jax.random.split(jax.random.key(0), 3)
    shape = CONFIG['shape']
    x = jax.random.uniform(kx, shape, dtype)
    mu = jax.random.uniform(kmu, shape, dtype)
    theta = jax.random.uniform(kth, shape, dtype)
    return x, mu, theta


def workload(x, mu, theta):
    eps = CONFIG['eps']
    if theta.ndim == 1:
        theta = theta.reshape(1, theta.shape[0])
    log_theta_mu_eps = jnp.log(theta + mu + eps)
    res = (theta * (jnp.log(theta + eps) - log_theta_mu_eps)
           + x * (jnp.log(mu + eps) - log_theta_mu_eps)
           + lax.lgamma(x + theta)
           - lax.lgamma(theta)
           - lax.lgamma(x + 1.0))
    if CONFIG['reduction'] == 'mean':
        return jnp.mean(res)
    if CONFIG['reduction'] == 'sum':
        return jnp.sum(res)
    raise NotImplementedError(f"reduction method {CONFIG['reduction']} is not implemented.")
