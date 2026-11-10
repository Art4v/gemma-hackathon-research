"""ParamSum: optional bilinear resize to the largest spatial shape, then per-channel weighted sum a*x + b*y (NCHW)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'ParamSum_13586',
    'batch': 16,
    'channels': 128,
    'height': 128,
    'width': 128,
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    kx, ky = jax.random.split(key)
    shape = (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width'])
    x = jax.random.uniform(kx, shape, dtype)
    y = jax.random.uniform(ky, shape, dtype)
    a = jnp.ones((CONFIG['channels'],), dtype)
    b = jnp.ones((CONFIG['channels'],), dtype)
    return x, y, a, b


def workload(x, y, a, b):
    bsize = x.shape[0]
    if x.shape[2:] > y.shape[2:]:
        y = jax.image.resize(y, y.shape[:2] + x.shape[2:], method='bilinear', antialias=False)
    elif x.shape[2:] < y.shape[2:]:
        x = jax.image.resize(x, x.shape[:2] + y.shape[2:], method='bilinear', antialias=False)
    a_expanded = jnp.broadcast_to(a, (bsize, a.shape[0]))[:, :, None, None]
    b_expanded = jnp.broadcast_to(b, (bsize, b.shape[0]))[:, :, None, None]
    return a_expanded * x + b_expanded * y
