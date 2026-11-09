"""[bf16 variant] LocalVariation: reflection pad -> 5x5 avg pool -> local variance of y (E[y^2] - E[y]^2)."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'LocalVariation_10879',
    'batch': 8,
    'channels': 16,
    'height': 512,
    'width': 512,
    'k_size': 5,
}


def create_inputs(dtype=jnp.bfloat16):
    kx, ky = jax.random.split(jax.random.key(0))
    shape = (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width'])
    x = jax.random.uniform(kx, shape, dtype)
    y = jax.random.uniform(ky, shape, dtype)
    return x, y


def _avg_pool(a, k):
    s = lax.reduce_window(a, jnp.array(0, a.dtype), lax.add,
                          (1, 1, k, k), (1, 1, 1, 1), 'VALID')
    return s / (k * k)


def workload(x, y):
    # ponytail: x only feeds sigma_x, which the PyTorch forward discards; output is sigma_y.
    k = CONFIG['k_size']
    p = k // 2
    yp = jnp.pad(y, ((0, 0), (0, 0), (p, p), (p, p)), mode='reflect')
    mu_y = _avg_pool(yp, k)
    return _avg_pool(yp * yp, k) - mu_y * mu_y
