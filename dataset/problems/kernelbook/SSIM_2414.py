"""SSIM loss: reflection pad -> 3x3 avg pools of x, y, x^2, y^2, xy -> SSIM map -> clamp((1-SSIM)/2, 0, 1)."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'SSIM_2414',
    'batch': 16,
    'channels': 4,
    'height': 512,
    'width': 512,
    'pool_size': 3,
    'pad': 1,
    'C1': 0.01 ** 2,
    'C2': 0.03 ** 2,
}


def create_inputs(dtype=jnp.float32):
    kx, ky = jax.random.split(jax.random.key(0))
    shape = (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width'])
    x = jax.random.uniform(kx, shape, dtype)
    y = jax.random.uniform(ky, shape, dtype)
    return x, y


def _avg_pool(a):
    k = CONFIG['pool_size']
    s = lax.reduce_window(a, jnp.array(0, a.dtype), lax.add, (1, 1, k, k), (1, 1, 1, 1), 'VALID')
    return s / (k * k)


def workload(x, y):
    p = CONFIG['pad']
    pad = ((0, 0), (0, 0), (p, p), (p, p))
    x = jnp.pad(x, pad, mode='reflect')
    y = jnp.pad(y, pad, mode='reflect')
    c1, c2 = CONFIG['C1'], CONFIG['C2']
    mu_x = _avg_pool(x)
    mu_y = _avg_pool(y)
    sigma_x = _avg_pool(x * x) - mu_x ** 2
    sigma_y = _avg_pool(y * y) - mu_y ** 2
    sigma_xy = _avg_pool(x * y) - mu_x * mu_y
    ssim_n = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
    ssim_d = (mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x + sigma_y + c2)
    return jnp.clip((1 - ssim_n / ssim_d) / 2, 0, 1)
