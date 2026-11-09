"""[bf16 variant] Discriminator2: shared nn.Bilinear(n_h, n_h, 1) on (h_pl, c) and (h_mi, c), squeeze(dim 2), concat on dim 1."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'Discriminator2_5996',
    'batch': 8,
    'dim_a': 128,
    'dim_c': 128,
    'n_h': 256,
    'out_features': 1,
    'squeeze_dim': 2,
    'concat_dim': 1,
}


def create_inputs(dtype=jnp.bfloat16):
    shape = (CONFIG['batch'], CONFIG['dim_a'], CONFIG['dim_c'], CONFIG['n_h'])
    k_c, k_pl, k_mi, k_w = jax.random.split(jax.random.key(0), 4)
    c = jax.random.uniform(k_c, shape, dtype)
    h_pl = jax.random.uniform(k_pl, shape, dtype)
    h_mi = jax.random.uniform(k_mi, shape, dtype)
    n_h, n_out = CONFIG['n_h'], CONFIG['out_features']
    weight = jax.random.normal(k_w, (n_out, n_h, n_h), dtype) * 0.02
    bias = jnp.zeros((n_out,), dtype)
    return c, h_pl, h_mi, weight, bias


def _bilinear(x1, x2, weight, bias):
    return jnp.einsum('...i,oij,...j->...o', x1, weight, x2) + bias


def _torch_squeeze(x, dim):
    # torch.squeeze(x, dim) is a no-op when that dim is not size 1.
    return jnp.squeeze(x, axis=dim) if x.shape[dim] == 1 else x


def workload(c, h_pl, h_mi, weight, bias):
    sq = CONFIG['squeeze_dim']
    sc_1 = _torch_squeeze(_bilinear(h_pl, c, weight, bias), sq)
    sc_2 = _torch_squeeze(_bilinear(h_mi, c, weight, bias), sq)
    return jnp.concatenate((sc_1, sc_2), axis=CONFIG['concat_dim'])
