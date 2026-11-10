"""Coboundary: batched matmul of x with D transpose, channel mix by theta, add bias."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'Coboundary_8862',
    'B': 32,
    'C_in': 256,
    'C_out': 256,
    'N': 2048,
    'M': 2048,
    'weight_scale': 0.02,
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    k_d, k_x, k_theta = jax.random.split(key, 3)
    D = jax.random.uniform(k_d, (CONFIG['N'], CONFIG['M']), dtype)
    x = jax.random.uniform(k_x, (CONFIG['B'], CONFIG['C_in'], CONFIG['M']), dtype)
    theta = jax.random.normal(k_theta, (CONFIG['C_out'], CONFIG['C_in']), dtype) * CONFIG['weight_scale']
    bias = jnp.zeros((1, CONFIG['C_out'], 1), dtype)
    return D, x, theta, bias


def workload(D, x, theta, bias):
    X = jnp.matmul(x, jnp.swapaxes(D, 0, 1))
    y = jnp.matmul(theta, X)
    return y + bias
