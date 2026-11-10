"""LayerNorm over the channel axis of (B, C, T): mean, residual, variance, normalize by sqrt(var + eps), affine scale and shift."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'LayerNorm_16764',
    'batch': 64,
    'num_channels': 1024,
    'timesteps': 1024,
    'eps': 1e-05,
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    x = jax.random.uniform(
        key,
        (CONFIG['batch'], CONFIG['num_channels'], CONFIG['timesteps']),
        dtype,
    )
    weight = jnp.ones((1, CONFIG['num_channels'], 1), dtype)
    bias = jnp.zeros((1, CONFIG['num_channels'], 1), dtype)
    return x, weight, bias


def workload(x, weight, bias):
    mu = jnp.mean(x, axis=1, keepdims=True)
    res_x = x - mu
    sigma = jnp.mean(res_x ** 2, axis=1, keepdims=True)
    out = res_x / jnp.sqrt(sigma + CONFIG['eps'])
    return out * weight + bias
