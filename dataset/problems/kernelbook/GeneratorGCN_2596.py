"""[bf16 variant] GeneratorGCN: three dense GCN layers, each x @ W, adj @ support, add bias, ReLU."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'GeneratorGCN_2596',
    'num_nodes': 4096,
    'nfeat': 1024,
    'nhid': 1024,
    'nclass': 1024,
    'weight_scale': 0.02,
}


def create_inputs(dtype=jnp.bfloat16):
    n = CONFIG['num_nodes']
    nfeat = CONFIG['nfeat']
    nhid = CONFIG['nhid']
    nclass = CONFIG['nclass']
    scale = CONFIG['weight_scale']
    keys = jax.random.split(jax.random.key(0), 8)
    x = jax.random.uniform(keys[0], (n, nfeat), dtype)
    adj = jax.random.uniform(keys[1], (n, n), dtype)
    w1 = jax.random.normal(keys[2], (nfeat, nhid), dtype) * scale
    b1 = jax.random.normal(keys[3], (nhid,), dtype) * scale
    w2 = jax.random.normal(keys[4], (nhid, nhid), dtype) * scale
    b2 = jax.random.normal(keys[5], (nhid,), dtype) * scale
    w3 = jax.random.normal(keys[6], (nhid, nclass), dtype) * scale
    b3 = jax.random.normal(keys[7], (nclass,), dtype) * scale
    return x, adj, w1, b1, w2, b2, w3, b3


def workload(x, adj, w1, b1, w2, b2, w3, b3):
    for w, b in ((w1, b1), (w2, b2), (w3, b3)):
        support = x @ w
        x = jax.nn.relu(adj @ support + b)
    return x
