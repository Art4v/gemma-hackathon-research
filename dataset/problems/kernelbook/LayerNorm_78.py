"""[bf16 variant] LayerNorm_78: per-sample flatten mean/std (ddof=1) normalization -> per-channel affine scale and shift."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'LayerNorm_78',
    'batch': 8,
    'num_features': 128,
    'height': 256,
    'width': 256,
    'eps': 1e-05,
    'ddof': 1,
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    x = jax.random.uniform(
        key,
        (CONFIG['batch'], CONFIG['num_features'], CONFIG['height'], CONFIG['width']),
        dtype,
    )
    gamma = jnp.ones((CONFIG['num_features'],), dtype)
    beta = jnp.zeros((CONFIG['num_features'],), dtype)
    return x, gamma, beta


def workload(x, gamma, beta):
    stat_shape = (-1,) + (1,) * (x.ndim - 1)
    flat = x.reshape(x.shape[0], -1)
    mean = jnp.mean(flat, axis=1).reshape(stat_shape)
    std = jnp.std(flat, axis=1, ddof=CONFIG['ddof']).reshape(stat_shape)
    y = (x - mean) / (std + CONFIG['eps'])
    param_shape = (1, -1) + (1,) * (x.ndim - 2)
    return y * gamma.reshape(param_shape) + beta.reshape(param_shape)
