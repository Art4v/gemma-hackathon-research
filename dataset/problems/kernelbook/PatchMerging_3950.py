"""PatchMerging (Swin3D): 2x2 spatial gather -> concat(4C) -> LayerNorm -> Linear(4C->2C, no bias)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'PatchMerging_3950',
    'batch': 8,
    'depth': 8,
    'height': 64,
    'width': 64,
    'dim': 128,
    'eps': 1e-5,
}


def create_inputs(dtype=jnp.float32):
    c = CONFIG
    k_x, k_w = jax.random.split(jax.random.key(0))
    x = jax.random.uniform(
        k_x, (c['batch'], c['depth'], c['height'], c['width'], c['dim']), dtype)
    gamma = jnp.ones((4 * c['dim'],), dtype)
    beta = jnp.zeros((4 * c['dim'],), dtype)
    w = jax.random.normal(k_w, (4 * c['dim'], 2 * c['dim']), dtype) * 0.02
    return x, gamma, beta, w


def workload(x, gamma, beta, w):
    h, wd = x.shape[2], x.shape[3]
    if h % 2 == 1 or wd % 2 == 1:
        x = jnp.pad(x, ((0, 0), (0, 0), (0, h % 2), (0, wd % 2), (0, 0)))
    x0 = x[:, :, 0::2, 0::2, :]
    x1 = x[:, :, 1::2, 0::2, :]
    x2 = x[:, :, 0::2, 1::2, :]
    x3 = x[:, :, 1::2, 1::2, :]
    y = jnp.concatenate([x0, x1, x2, x3], axis=-1)
    mean = jnp.mean(y, axis=-1, keepdims=True)
    var = jnp.mean(jnp.square(y - mean), axis=-1, keepdims=True)
    y = (y - mean) * jax.lax.rsqrt(var + CONFIG['eps']) * gamma + beta
    return y @ w
