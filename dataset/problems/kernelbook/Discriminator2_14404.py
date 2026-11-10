"""[bf16 variant] Discriminator2: shared bilinear(h_pl, c) and bilinear(h_mi, c) -> conditional squeeze(axis=2) -> concat(axis=1)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'Discriminator2_14404',
    'batch': 8,
    'dim_1': 128,
    'dim_2': 128,
    'n_h': 128,
    'n_out': 1,
    'squeeze_axis': 2,
    'concat_axis': 1,
}


def create_inputs(dtype=jnp.bfloat16):
    shape = (CONFIG['batch'], CONFIG['dim_1'], CONFIG['dim_2'], CONFIG['n_h'])
    k_c, k_pl, k_mi, k_w = jax.random.split(jax.random.key(0), 4)
    c = jax.random.uniform(k_c, shape, dtype)
    h_pl = jax.random.uniform(k_pl, shape, dtype)
    h_mi = jax.random.uniform(k_mi, shape, dtype)
    weight = jax.random.normal(k_w, (CONFIG['n_out'], CONFIG['n_h'], CONFIG['n_h']), dtype) * 0.02
    bias = jnp.zeros((CONFIG['n_out'],), dtype)
    return c, h_pl, h_mi, weight, bias


def _bilinear(x1, x2, weight, bias):
    x1_w = jnp.einsum('...i,kij->...kj', x1, weight)
    return jnp.einsum('...kj,...j->...k', x1_w, x2) + bias


def _squeeze_if_singleton(x, axis):
    return jnp.squeeze(x, axis) if x.shape[axis] == 1 else x


def workload(c, h_pl, h_mi, weight, bias):
    sc_1 = _squeeze_if_singleton(_bilinear(h_pl, c, weight, bias), CONFIG['squeeze_axis'])
    sc_2 = _squeeze_if_singleton(_bilinear(h_mi, c, weight, bias), CONFIG['squeeze_axis'])
    return jnp.concatenate((sc_1, sc_2), axis=CONFIG['concat_axis'])
