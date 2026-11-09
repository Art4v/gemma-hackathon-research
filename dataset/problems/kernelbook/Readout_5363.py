"""[bf16 variant] Readout: concat -> Linear+ReLU -> Linear+ReLU -> Linear -> mean over nodes."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'Readout_5363',
    'num_nodes': 65536,
    'attr_dim': 512,
    'embedding_dim': 512,
    'hidden_dim': 1024,
    'output_dim': 512,
    'num_cats': 128,
}


def create_inputs(dtype=jnp.bfloat16):
    c = CONFIG
    ks = jax.random.split(jax.random.key(0), 8)
    feats = jax.random.uniform(ks[0], (c['num_nodes'], c['attr_dim']), dtype)
    embs = jax.random.uniform(ks[1], (c['num_nodes'], c['embedding_dim']), dtype)
    in1 = c['attr_dim'] + c['embedding_dim']
    w1 = jax.random.normal(ks[2], (in1, c['hidden_dim']), dtype) * 0.02
    b1 = jax.random.normal(ks[3], (c['hidden_dim'],), dtype) * 0.02
    w2 = jax.random.normal(ks[4], (c['hidden_dim'], c['output_dim']), dtype) * 0.02
    b2 = jax.random.normal(ks[5], (c['output_dim'],), dtype) * 0.02
    w3 = jax.random.normal(ks[6], (c['output_dim'], c['num_cats']), dtype) * 0.02
    b3 = jax.random.normal(ks[7], (c['num_cats'],), dtype) * 0.02
    return feats, embs, w1, b1, w2, b2, w3, b3


def workload(node_features, node_embeddings, w1, b1, w2, b2, w3, b3):
    x = jnp.concatenate((node_features, node_embeddings), axis=1)
    h = jax.nn.relu(x @ w1 + b1)
    g = jax.nn.relu(h @ w2 + b2)
    return jnp.mean(g @ w3 + b3, axis=0)
