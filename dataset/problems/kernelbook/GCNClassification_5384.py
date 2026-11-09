"""[bf16 variant] GCN (two dense adj-matmul layers with ReLU) + concat + 3-layer MLP readout + mean over nodes."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'GCNClassification_5384',
    'num_nodes': 4096,
    'feat_dim': 1024,
    'hidden_dim': 1024,
    'node_embedding_dim': 1024,
    'readout_hidden_dim': 1024,
    'graph_embedding_dim': 512,
    'num_categories': 128,
}


def create_inputs(dtype=jnp.bfloat16):
    c = CONFIG
    n, f = c['num_nodes'], c['feat_dim']
    h1, h2 = c['hidden_dim'], c['node_embedding_dim']
    rh, ge, nc = c['readout_hidden_dim'], c['graph_embedding_dim'], c['num_categories']
    ks = jax.random.split(jax.random.key(0), 10)
    adj = jax.random.uniform(ks[0], (n, n), dtype)
    x = jax.random.uniform(ks[1], (n, f), dtype)
    w1 = jax.random.normal(ks[2], (f, h1), dtype) * 0.02
    w2 = jax.random.normal(ks[3], (h1, h2), dtype) * 0.02
    l1_w = jax.random.normal(ks[4], (f + h2, rh), dtype) * 0.02
    l1_b = jax.random.normal(ks[5], (rh,), dtype) * 0.02
    l2_w = jax.random.normal(ks[6], (rh, ge), dtype) * 0.02
    l2_b = jax.random.normal(ks[7], (ge,), dtype) * 0.02
    out_w = jax.random.normal(ks[8], (ge, nc), dtype) * 0.02
    out_b = jax.random.normal(ks[9], (nc,), dtype) * 0.02
    return adj, x, w1, w2, l1_w, l1_b, l2_w, l2_b, out_w, out_b


def workload(adj, x, w1, w2, l1_w, l1_b, l2_w, l2_b, out_w, out_b):
    # GCN; dropout is identity in inference.
    emb = jax.nn.relu(adj @ (x @ w1))
    emb = jax.nn.relu(adj @ (emb @ w2))
    # Readout.
    combined = jnp.concatenate((x, emb), axis=1)
    hidden = jax.nn.relu(combined @ l1_w + l1_b)
    graph = jax.nn.relu(hidden @ l2_w + l2_b)
    return jnp.mean(graph @ out_w + out_b, axis=0)
