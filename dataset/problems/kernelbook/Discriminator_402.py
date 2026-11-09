"""[bf16 variant] Discriminator_402: shared nn.Bilinear(n_h, n_h, 1) scoring of (h_pl, c) and (h_mi, c), squeeze, concat."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'Discriminator_402',
    'batch': 8,
    'num_nodes': 4096,
    'n_h': 512,
    'out_features': 1,
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    k_c, k_pl, k_mi, k_w = jax.random.split(key, 4)
    b, n, h = CONFIG['batch'], CONFIG['num_nodes'], CONFIG['n_h']
    o = CONFIG['out_features']
    c = jax.random.uniform(k_c, (n,), dtype)
    h_pl = jax.random.uniform(k_pl, (b, n, h), dtype)
    h_mi = jax.random.uniform(k_mi, (b, n, h), dtype)
    w = jax.random.normal(k_w, (o, h, h), dtype) * 0.02
    bias = jnp.zeros((o,), dtype)
    return c, h_pl, h_mi, w, bias


def _bilinear(x1, x2, w, bias):
    # out[b, n, o] = x1[b, n, :] @ w[o] @ x2[b, n, :] + bias[o]
    t = jnp.einsum('bnj,ojk->bnok', x1, w)
    return jnp.sum(t * x2[:, :, None, :], axis=-1) + bias


def workload(c, h_pl, h_mi, w, bias):
    # torch: unsqueeze(c, 1) -> (N, 1), expand_as right-aligns to (B, N, n_h).
    c_x = jnp.broadcast_to(c[:, None], h_pl.shape)
    sc_1 = jnp.squeeze(_bilinear(h_pl, c_x, w, bias), axis=2)
    sc_2 = jnp.squeeze(_bilinear(h_mi, c_x, w, bias), axis=2)
    return jnp.concatenate((sc_1, sc_2), axis=1)
