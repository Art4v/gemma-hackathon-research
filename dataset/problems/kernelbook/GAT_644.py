"""[bf16 variant] GAT: two GAT layers (linear, additive attention scaled by M, leaky_relu, adj mask, softmax, aggregate, elu), L2 normalize, sigmoid(Z Z^T) decode."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'GAT_644',
    'num_nodes': 4096,
    'num_features': 512,
    'hidden_size': 256,
    'embedding_size': 128,
    'alpha': 4.0,
    'mask_value': -9e15,
    'norm_eps': 1e-12,
}


def create_inputs(dtype=jnp.bfloat16):
    n = CONFIG['num_nodes']
    f = CONFIG['num_features']
    hdim = CONFIG['hidden_size']
    e = CONFIG['embedding_size']
    ks = jax.random.split(jax.random.key(0), 9)
    x = jax.random.uniform(ks[0], (n, f), dtype)
    adj = jax.random.uniform(ks[1], (n, n), dtype)
    m = jax.random.uniform(ks[2], (n, n), dtype)
    w1 = jax.random.normal(ks[3], (f, hdim), dtype) * 0.02
    a_self1 = jax.random.normal(ks[4], (hdim, 1), dtype) * 0.02
    a_neighs1 = jax.random.normal(ks[5], (hdim, 1), dtype) * 0.02
    w2 = jax.random.normal(ks[6], (hdim, e), dtype) * 0.02
    a_self2 = jax.random.normal(ks[7], (e, 1), dtype) * 0.02
    a_neighs2 = jax.random.normal(ks[8], (e, 1), dtype) * 0.02
    return x, adj, m, w1, a_self1, a_neighs1, w2, a_self2, a_neighs2


def _gat_layer(x, adj, m, w, a_self, a_neighs):
    h = x @ w
    attn = (h @ a_self) + (h @ a_neighs).T
    attn = jax.nn.leaky_relu(attn * m, negative_slope=CONFIG['alpha'])
    attn = jnp.where(adj > 0, attn, jnp.asarray(CONFIG['mask_value'], attn.dtype))
    attention = jax.nn.softmax(attn, axis=1)
    return jax.nn.elu(attention @ h)


def workload(x, adj, m, w1, a_self1, a_neighs1, w2, a_self2, a_neighs2):
    h = _gat_layer(x, adj, m, w1, a_self1, a_neighs1)
    h = _gat_layer(h, adj, m, w2, a_self2, a_neighs2)
    norm = jnp.sqrt(jnp.sum(h * h, axis=1, keepdims=True))
    z = h / jnp.maximum(norm, CONFIG['norm_eps'])
    a_pred = jax.nn.sigmoid(z @ z.T)
    return a_pred, z
