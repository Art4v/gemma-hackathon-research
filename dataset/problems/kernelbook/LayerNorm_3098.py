"""LayerNorm_3098: mean/unbiased-std over axis 2, (x - mu) / (std + eps) * alpha + beta."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'LayerNorm_3098',
    'batch': 8,
    'channels': 64,
    'length': 256,
    'hidden_size': 512,
    'reduce_axis': 2,
    'eps': 1e-4,
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    shape = (CONFIG['batch'], CONFIG['channels'], CONFIG['length'], CONFIG['hidden_size'])
    x = jax.random.uniform(key, shape, dtype)
    alpha = jnp.ones((1, 1, CONFIG['hidden_size']), dtype)
    beta = jnp.zeros((1, 1, CONFIG['hidden_size']), dtype)
    return x, alpha, beta


def workload(x, alpha, beta):
    axis = CONFIG['reduce_axis']
    mu = jnp.mean(x, axis=axis, keepdims=True)
    # torch.std defaults to the unbiased estimator (ddof=1)
    sigma = jnp.std(x, axis=axis, keepdims=True, ddof=1)
    return (x - mu) / (sigma + CONFIG['eps']) * alpha + beta
