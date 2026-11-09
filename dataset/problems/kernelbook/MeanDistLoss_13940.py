"""MeanDistLoss: mean(cdist(x, y, p=2)) over batched 4D inputs."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'MeanDistLoss_13940',
    'batch': 8,
    'heads': 16,
    'p_rows': 1024,
    'r_rows': 1024,
    'dim': 128,
    'p': 2,
}


def create_inputs(dtype=jnp.float32):
    kx, ky = jax.random.split(jax.random.key(0))
    b, h, m = CONFIG['batch'], CONFIG['heads'], CONFIG['dim']
    x = jax.random.uniform(kx, (b, h, CONFIG['p_rows'], m), dtype)
    y = jax.random.uniform(ky, (b, h, CONFIG['r_rows'], m), dtype)
    return x, y


def workload(x, y):
    # Matches torch's mm-based euclidean cdist path (used when P or R > 25).
    x_sq = jnp.sum(x * x, axis=-1, keepdims=True)
    y_sq = jnp.sum(y * y, axis=-1, keepdims=True)
    xy = jnp.einsum('bhpm,bhrm->bhpr', x, y, precision=lax.Precision.HIGHEST)
    d2 = x_sq + jnp.swapaxes(y_sq, -1, -2) - 2.0 * xy
    dist = jnp.sqrt(jnp.maximum(d2, 0.0))
    return jnp.mean(dist)
