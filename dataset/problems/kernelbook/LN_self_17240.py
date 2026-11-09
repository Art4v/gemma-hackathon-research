"""[bf16 variant] LN_self: per-sample LayerNorm over (C, H, W) with biased variance, then per-channel affine gamma * x + beta."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'LN_self_17240',
    'batch': 8,
    'num_features': 128,
    'height': 256,
    'width': 256,
    'eps': 1e-5,
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    shape = (CONFIG['batch'], CONFIG['num_features'], CONFIG['height'], CONFIG['width'])
    x = jax.random.uniform(key, shape, dtype)
    param_shape = (1, CONFIG['num_features'], 1, 1)
    gamma = jnp.ones(param_shape, dtype)
    beta = jnp.zeros(param_shape, dtype)
    return x, gamma, beta


def workload(x, gamma, beta):
    axes = (1, 2, 3)
    mean = jnp.mean(x, axis=axes, keepdims=True)
    var = jnp.mean(jnp.square(x - mean), axis=axes, keepdims=True)
    y = (x - mean) / jnp.sqrt(var + CONFIG['eps'])
    return gamma * y + beta
