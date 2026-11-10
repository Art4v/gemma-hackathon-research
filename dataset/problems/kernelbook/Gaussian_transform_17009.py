"""Gaussian point transform: expand axis, subtract t_param, square, sum over axis 1, scale by 2*sigma^2, exp(-)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'Gaussian_transform_17009',
    'batch': 8,
    'channels': 128,
    'height': 256,
    'width': 256,
    'output_dim': 256,
    'sigma_size': 1,
    't_scale': 0.02,
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    k_x, k_t = jax.random.split(key)
    x = jax.random.uniform(
        k_x,
        (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width']),
        dtype,
    )
    t_param = jax.random.normal(k_t, (CONFIG['output_dim'],), dtype) * CONFIG['t_scale']
    sigma = jnp.ones((CONFIG['sigma_size'],), dtype)
    return x, t_param, sigma


def workload(x, t_param, sigma):
    squared_distance = jnp.sum((x[:, :, None] - t_param) ** 2, axis=1)
    return jnp.exp(-squared_distance / (2 * sigma ** 2))
