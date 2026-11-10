"""Encoding layer: scaled L2 distance to codewords -> softmax over codes -> residual aggregation."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'Encoding_5325',
    'batch': 16,
    'channels': 512,
    'height': 64,
    'width': 64,
    'num_codes': 32,
}


def create_inputs(dtype=jnp.float32):
    k_x, k_c, k_s = jax.random.split(jax.random.key(0), 3)
    x = jax.random.uniform(
        k_x, (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width']), dtype)
    codewords = jax.random.normal(k_c, (CONFIG['num_codes'], CONFIG['channels']), dtype) * 0.02
    # scale in [-1, 0) like the module init: negative scale makes softmax favour near codes.
    scale = jax.random.uniform(k_s, (CONFIG['num_codes'],), dtype, minval=-1.0, maxval=0.0)
    return x, codewords, scale


def workload(x, codewords, scale):
    b, c = x.shape[0], x.shape[1]
    hp = lax.Precision.HIGHEST
    xf = jnp.transpose(x.reshape(b, c, -1), (0, 2, 1))  # (B, N, C)
    # ponytail: ||x-c||^2 expanded as x^2 - 2xc + c^2; the literal (B,N,K,C) broadcast is ~4 GB.
    x_sq = jnp.sum(xf * xf, axis=-1, keepdims=True)  # (B, N, 1)
    c_sq = jnp.sum(codewords * codewords, axis=-1)  # (K,)
    xc = jnp.einsum('bnc,kc->bnk', xf, codewords, precision=hp)
    dist = x_sq - 2.0 * xc + c_sq[None, None, :]
    weights = jax.nn.softmax(scale[None, None, :] * dist, axis=2)  # (B, N, K)
    # sum_n a[b,n,k] * (x[b,n,:] - c[k,:]) = a^T x - (sum_n a) * c
    ax = jnp.einsum('bnk,bnc->bkc', weights, xf, precision=hp)
    a_sum = jnp.sum(weights, axis=1)  # (B, K)
    return ax - a_sum[:, :, None] * codewords[None, :, :]
