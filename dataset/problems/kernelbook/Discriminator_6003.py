"""[bf16 variant] Discriminator: expand context c over nodes, shared nn.Bilinear(n_h, n_h, 1) on (h_pl, c) and (h_mi, c), concat scores."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'Discriminator_6003',
    'batch': 16,
    'n_nodes': 4096,
    'n_h': 512,
    'out_features': 1,
}


def create_inputs(dtype=jnp.bfloat16):
    b, n, h = CONFIG['batch'], CONFIG['n_nodes'], CONFIG['n_h']
    o = CONFIG['out_features']
    k_c, k_pl, k_mi, k_w = jax.random.split(jax.random.key(0), 4)
    c = jax.random.uniform(k_c, (b, h), dtype)
    h_pl = jax.random.uniform(k_pl, (b, n, h), dtype)
    h_mi = jax.random.uniform(k_mi, (b, n, h), dtype)
    weight = jax.random.normal(k_w, (o, h, h), dtype) * 0.02
    bias = jnp.zeros((o,), dtype)
    return c, h_pl, h_mi, weight, bias


def _bilinear_score(x1, x2, weight, bias):
    # Out features == 1, so the squeeze(2) in torch is the [0] index here.
    return jnp.sum(jnp.matmul(x1, weight[0]) * x2, axis=-1) + bias[0]


def workload(c, h_pl, h_mi, weight, bias):
    c_x = jnp.broadcast_to(jnp.expand_dims(c, 1), h_pl.shape)
    sc_1 = _bilinear_score(h_pl, c_x, weight, bias)
    sc_2 = _bilinear_score(h_mi, c_x, weight, bias)
    return jnp.concatenate([sc_1, sc_2], axis=1)
