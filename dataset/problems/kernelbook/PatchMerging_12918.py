"""PatchMerging: 2x2 strided gather + concat -> LayerNorm(4C) -> Linear(4C -> 2C, no bias)."""
import math

import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'PatchMerging_12918',
    'batch': 8,
    'height': 128,
    'width': 128,
    'dim': 256,
    'eps': 1e-5,
}


def create_inputs(dtype=jnp.float32):
    kx, kw = jax.random.split(jax.random.key(0))
    B, H, W, C = CONFIG['batch'], CONFIG['height'], CONFIG['width'], CONFIG['dim']
    x = jax.random.uniform(kx, (B, H * W, C), dtype)
    gamma = jnp.ones((4 * C,), dtype)
    beta = jnp.zeros((4 * C,), dtype)
    w = jax.random.normal(kw, (4 * C, 2 * C), dtype) * 0.02
    return x, gamma, beta, w


def workload(x, gamma, beta, w):
    B, L, C = x.shape
    H = math.isqrt(L)
    W = H
    x = x.reshape(B, H, W, C)
    if H % 2 == 1 or W % 2 == 1:
        x = jnp.pad(x, ((0, 0), (0, H % 2), (0, W % 2), (0, 0)))
    x0 = x[:, 0::2, 0::2, :]
    x1 = x[:, 1::2, 0::2, :]
    x2 = x[:, 0::2, 1::2, :]
    x3 = x[:, 1::2, 1::2, :]
    x = jnp.concatenate([x0, x1, x2, x3], axis=-1).reshape(B, -1, 4 * C)
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.mean(jnp.square(x - mean), axis=-1, keepdims=True)
    x = (x - mean) * jax.lax.rsqrt(var + CONFIG['eps']) * gamma + beta
    return x @ w
