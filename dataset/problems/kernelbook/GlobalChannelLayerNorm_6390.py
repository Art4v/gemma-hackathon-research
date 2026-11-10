"""[bf16 variant] Global channel layer norm: per-sample mean/variance over (C, T), normalize, then elementwise gamma/beta affine."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'GlobalChannelLayerNorm_6390',
    'batch': 128,
    'channels': 512,
    'time': 1024,
    'eps': 1e-05,
    'norm_axes': (1, 2),
    'affine_trailing_dim': 1,
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    x = jax.random.uniform(
        key, (CONFIG['batch'], CONFIG['channels'], CONFIG['time']), dtype
    )
    gamma = jnp.ones((CONFIG['channels'], CONFIG['affine_trailing_dim']), dtype)
    beta = jnp.zeros((CONFIG['channels'], CONFIG['affine_trailing_dim']), dtype)
    return x, gamma, beta


def workload(x, gamma, beta):
    axes = CONFIG['norm_axes']
    mean = jnp.mean(x, axis=axes, keepdims=True)
    var = jnp.mean((x - mean) ** 2, axis=axes, keepdims=True)
    return gamma * (x - mean) / jnp.sqrt(var + CONFIG['eps']) + beta
