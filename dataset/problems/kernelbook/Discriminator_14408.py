"""[bf16 variant] Discriminator_14408: shared nn.Bilinear(n_h, n_h, 1) scoring of (h_pl, c) and (h_mi, c), squeeze, concat."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'Discriminator_14408',
    'batch': 8,
    'n_nodes': 8192,
    'n_h': 512,
}


def create_inputs(dtype=jnp.bfloat16):
    b, n, h = CONFIG['batch'], CONFIG['n_nodes'], CONFIG['n_h']
    k_c, k_pl, k_mi, k_w = jax.random.split(jax.random.key(0), 4)
    c = jax.random.uniform(k_c, (b, h), dtype)
    h_pl = jax.random.uniform(k_pl, (b, n, h), dtype)
    h_mi = jax.random.uniform(k_mi, (b, n, h), dtype)
    w = jax.random.normal(k_w, (h, h), dtype) * 0.02
    bias = jnp.zeros((1,), dtype)
    return c, h_pl, h_mi, w, bias


def _bilinear(x, y, w, bias):
    # nn.Bilinear with out_features=1: sum_ij x_i W_ij y_j + b
    return jnp.sum(jnp.matmul(x, w) * y, axis=-1) + bias[0]


def workload(c, h_pl, h_mi, w, bias):
    # Mirrors torch unsqueeze(1).expand_as(h_pl) for any c rank.
    c_x = jnp.broadcast_to(jnp.expand_dims(c, 1), h_pl.shape)
    sc_1 = _bilinear(h_pl, c_x, w, bias)
    sc_2 = _bilinear(h_mi, c_x, w, bias)
    return jnp.concatenate((sc_1, sc_2), axis=1)
