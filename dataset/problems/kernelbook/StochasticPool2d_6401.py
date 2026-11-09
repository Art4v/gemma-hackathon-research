"""[bf16 variant] StochasticPool2d (training path): zero pad (right/bottom) -> max_pool2d(k=2, s=1) -> gather random row/col per grid cell."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'StochasticPool2d_6401',
    'batch': 16,
    'channels': 64,
    'height': 256,
    'width': 256,
    'kernel_size': 2,
    'grid_size': 2,
}


def _cell_indices(key, size):
    g = CONFIG['grid_size']
    n = size // g
    offsets = g * jnp.arange(n, dtype=jnp.int32)
    # Last cell may be wider when size is not a multiple of grid (torch: max_range = size - offset).
    max_range = jnp.full((n,), g, dtype=jnp.int32).at[-1].set(size - g * (n - 1))
    r = jax.random.randint(key, (n,), 0, max_range, dtype=jnp.int32)
    return offsets + r


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    kx, kh, kw = jax.random.split(key, 3)
    x = jax.random.uniform(
        kx, (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width']), dtype)
    h_indices = _cell_indices(kh, CONFIG['height'])
    w_indices = _cell_indices(kw, CONFIG['width'])
    return x, h_indices, w_indices


def workload(x, h_indices, w_indices):
    k = CONFIG['kernel_size']
    xp = jnp.pad(x, ((0, 0), (0, 0), (0, 1), (0, 1)))
    pooled = lax.reduce_window(
        xp, -jnp.inf, lax.max, (1, 1, k, k), (1, 1, 1, 1), 'VALID')
    out = jnp.take(pooled, h_indices, axis=2)
    return jnp.take(out, w_indices, axis=3)
