"""[bf16 variant] Encoding layer: flatten NCHW -> scaled L2 to codewords -> softmax over codes -> residual aggregation."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'Encoding_2367',
    'batch': 8,
    'channels': 512,
    'height': 64,
    'width': 64,
    'num_codes': 32,
}


def create_inputs(dtype=jnp.bfloat16):
    k_x, k_cw, k_s = jax.random.split(jax.random.key(0), 3)
    b, c, h, w, k = (CONFIG['batch'], CONFIG['channels'], CONFIG['height'],
                     CONFIG['width'], CONFIG['num_codes'])
    x = jax.random.uniform(k_x, (b, c, h, w), dtype)
    std = 1.0 / (k * c) ** 0.5
    codewords = jax.random.uniform(k_cw, (k, c), dtype, minval=-std, maxval=std)
    scale = jax.random.uniform(k_s, (k,), dtype, minval=-1.0, maxval=0.0)
    return x, codewords, scale


def workload(x, codewords, scale):
    b, c = x.shape[0], x.shape[1]
    xf = jnp.transpose(x.reshape(b, c, -1), (0, 2, 1))  # (B, N, C)
    # Direct form; XLA fuses broadcast-sub-square-reduce, no (B,N,K,C) buffer.
    dist = jnp.sum((xf[:, :, None, :] - codewords[None, None, :, :]) ** 2, axis=3)
    a = jax.nn.softmax(scale[None, None, :] * dist, axis=2)  # (B, N, K)
    # sum_n a_nk (x_n - c_k) = a^T x - (sum_n a_nk) c_k
    ax = jnp.einsum('bnk,bnc->bkc', a, xf, precision=jax.lax.Precision.HIGHEST)
    return ax - jnp.sum(a, axis=1)[:, :, None] * codewords[None, :, :]
