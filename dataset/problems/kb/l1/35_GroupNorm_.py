"""GroupNorm: per-(sample, group) mean/var normalization over (C/G, H, W) followed by per-channel affine scale/shift."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': '35_GroupNorm_',
    'batch_size': 8,
    'features': 128,
    'num_groups': 8,
    'dim1': 256,
    'dim2': 256,
    'eps': 1e-5,
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    x = jax.random.uniform(
        key,
        (CONFIG['batch_size'], CONFIG['features'], CONFIG['dim1'], CONFIG['dim2']),
        dtype,
    )
    gamma = jnp.ones((CONFIG['features'],), dtype)
    beta = jnp.zeros((CONFIG['features'],), dtype)
    return (x, gamma, beta)


def workload(x, gamma, beta):
    n, c = x.shape[0], x.shape[1]
    g = CONFIG['num_groups']
    eps = CONFIG['eps']
    xg = x.reshape(n, g, -1)
    mean = jnp.mean(xg, axis=-1, keepdims=True)
    xc = xg - mean
    var = jnp.mean(xc * xc, axis=-1, keepdims=True)
    inv = jax.lax.rsqrt(var + eps)
    y = (xc * inv).reshape(x.shape)
    bshape = (1, c) + (1,) * (x.ndim - 2)
    return y * gamma.reshape(bshape) + beta.reshape(bshape)
