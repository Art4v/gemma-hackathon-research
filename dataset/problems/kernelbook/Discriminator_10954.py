"""Discriminator: shared nn.Bilinear(n_h, n_h, 1) scoring of (h_pl, c) and (h_mi, c), squeeze, concat on dim 1, slice first half."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'Discriminator_10954',
    'batch': 16,
    'dim1': 256,
    'dim2': 256,
    'n_h': 128,
    'out_features': 1,
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    k_c, k_pl, k_mi, k_w = jax.random.split(key, 4)
    shape = (CONFIG['batch'], CONFIG['dim1'], CONFIG['dim2'], CONFIG['n_h'])
    c = jax.random.uniform(k_c, shape, dtype)
    h_pl = jax.random.uniform(k_pl, shape, dtype)
    h_mi = jax.random.uniform(k_mi, shape, dtype)
    w = jax.random.normal(k_w, (CONFIG['out_features'], CONFIG['n_h'], CONFIG['n_h']), dtype) * 0.02
    b = jnp.zeros((CONFIG['out_features'],), dtype)
    return c, h_pl, h_mi, w, b


def _bilinear(x1, x2, w, b):
    # PyTorch Bilinear: y_o = sum_ij x1_i * W[o, i, j] * x2_j + b_o
    y = jnp.einsum('...i,oij,...j->...o', x1, w, x2, precision=lax.Precision.HIGHEST)
    return y + b


def workload(c, h_pl, h_mi, w, b):
    sc_1 = jnp.squeeze(_bilinear(h_pl, c, w, b), axis=-1)
    sc_2 = jnp.squeeze(_bilinear(h_mi, c, w, b), axis=-1)
    logits = jnp.concatenate((sc_1, sc_2), axis=1)
    v = logits.shape[1]
    return logits, logits[:, :v // 2]
