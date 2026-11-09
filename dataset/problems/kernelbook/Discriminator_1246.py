"""Discriminator_1246: shared nn.Bilinear(n_h, n_h, 1) scores of (h_pl, c) and (h_mi, c), squeeze(1), concat on dim 0."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'Discriminator_1246',
    'batch': 16,
    'dim1': 16,
    'dim2': 128,
    'n_h': 512,
    'out_features': 1,
}


def create_inputs(dtype=jnp.float32):
    shape = (CONFIG['batch'], CONFIG['dim1'], CONFIG['dim2'], CONFIG['n_h'])
    k_c, k_pl, k_mi, k_w = jax.random.split(jax.random.key(0), 4)
    c = jax.random.uniform(k_c, shape, dtype)
    h_pl = jax.random.uniform(k_pl, shape, dtype)
    h_mi = jax.random.uniform(k_mi, shape, dtype)
    n_h, n_out = CONFIG['n_h'], CONFIG['out_features']
    w = jax.random.normal(k_w, (n_out, n_h, n_h), dtype) * 0.02
    b = jnp.zeros((n_out,), dtype)
    return c, h_pl, h_mi, w, b


def _bilinear(x1, x2, w, b):
    out = jnp.einsum('...i,oij,...j->...o', x1, w, x2, precision=lax.Precision.HIGHEST)
    return out + b


def _torch_squeeze1(x):
    # torch.squeeze(x, 1) only drops dim 1 when it has size 1.
    return jnp.squeeze(x, 1) if x.shape[1] == 1 else x


def workload(c, h_pl, h_mi, w, b):
    c_x = jnp.broadcast_to(c, h_pl.shape)
    sc_1 = _torch_squeeze1(_bilinear(h_pl, c_x, w, b))
    sc_2 = _torch_squeeze1(_bilinear(h_mi, c_x, w, b))
    return jnp.concatenate([sc_1, sc_2], axis=0)
