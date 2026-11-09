"""LayerNorm_6248: mean/unbiased-std over axis 2, (x - mu) / (sigma + eps) * alpha + beta (alpha/beta broadcast on last axis)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'LayerNorm_6248',
    'batch': 16,
    'channels': 64,
    'height': 128,
    'hidden_size': 512,
    'reduce_axis': 2,
    'eps': 1e-4,
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    shape = (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['hidden_size'])
    x = jax.random.uniform(key, shape, dtype)
    alpha = jnp.ones((1, 1, CONFIG['hidden_size']), dtype)
    beta = jnp.zeros((1, 1, CONFIG['hidden_size']), dtype)
    return x, alpha, beta


def workload(x, alpha, beta):
    axis = CONFIG['reduce_axis']
    mu = jnp.mean(x, axis=axis, keepdims=True)
    # torch.std defaults to the unbiased estimator; eps is added to sigma, not variance.
    sigma = jnp.std(x, axis=axis, keepdims=True, ddof=1)
    # alpha/beta are (1, 1, H); like torch expand_as they right-align and broadcast on the last axis.
    return (x - mu) / (sigma + CONFIG['eps']) * alpha + beta
