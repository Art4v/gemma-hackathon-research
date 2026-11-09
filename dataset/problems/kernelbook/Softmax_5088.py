"""[bf16 variant] Gaussian-propagating softmax: exp(mean + var/2) moments, sum over dim=1, normalize mean and variance."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'Softmax_5088',
    'batch': 8,
    'channels': 128,
    'height': 128,
    'width': 256,
    'dim': 1,
    'eps': 1e-5,
}


def create_inputs(dtype=jnp.bfloat16):
    k_mean, k_var = jax.random.split(jax.random.key(0))
    shape = (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width'])
    features_mean = jax.random.uniform(k_mean, shape, dtype)
    features_variance = jax.random.uniform(k_var, shape, dtype)
    return features_mean, features_variance


def workload(features_mean, features_variance):
    dim = CONFIG['dim']
    log_mean = features_mean + 0.5 * features_variance
    log_var = jnp.exp(2 * log_mean) * (jnp.exp(features_variance) - 1)
    log_mean = jnp.exp(log_mean)
    constant = jnp.sum(log_mean, axis=dim, keepdims=True) + CONFIG['eps']
    outputs_mean = log_mean / constant
    outputs_variance = log_var / constant ** 2
    return outputs_mean, outputs_variance
